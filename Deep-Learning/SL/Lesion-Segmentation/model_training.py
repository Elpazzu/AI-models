import zipfile, os, numpy as np, pickle, yaml, gc, tensorflow as tf
import segmentation_models as sm, albumentations as A
import tensorflow_addons as tfa
from ImageDataAugmentor.image_data_augmentor import *
from keras import backend as K
from segmentation_models import Unet, FPN
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config
config = load_config("utils/model_config_S1.yaml")

# del black background
def del_darkimg(img,mask):
    temp_i = []
    temp_m = []
    for i in tqdm(range(img.shape[0])):
        if np.sum(img[i])!=0:
            temp_i.append(img[i])
            temp_m.append(mask[i])
    temp = np.array(temp_i)
    temp2 = np.array(temp_m)
    return temp, temp2

# loading npy MRI dataset
def dataloader():
    # loading train data 3.0T: image / masks
    X_train = np.expand_dims(np.load(MRI_nii_folder_path + config["train_img_p3"]), axis=-1)
    y_train = np.expand_dims(np.load(MRI_nii_folder_path + config["train_msk_p3"]), axis=-1)
    # loading valida data 3.0T + 1.5T: image / masks
    X_valid = np.expand_dims(np.load(MRI_nii_folder_path + config["valid_img_p3"]), axis=-1)
    y_valid = np.expand_dims(np.load(MRI_nii_folder_path + config["valid_msk_p3"]), axis=-1)
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    # (140, 32, 384, 384, 1)
    X_train = np.reshape(X_train, (X_train.shape[0]*32,384,384,1))
    y_train = np.reshape(y_train, (y_train.shape[0]*32,384,384,1))
    X_valid = np.reshape(X_valid, (X_valid.shape[0]*32,384,384,1))
    y_valid = np.reshape(y_valid, (y_valid.shape[0]*32,384,384,1))
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    X_train, y_train = del_darkimg(X_train, y_train)
    X_valid, y_valid = del_darkimg(X_valid, y_valid)
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    return X_train.astype(np.float32), y_train.astype(np.int8), X_valid.astype(np.float32), y_valid.astype(np.int8)

# Setting initialization callback value and function
class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.001):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.lr = init_lr

    def get_callbacks(self, model_prefix='Model'):
        filepath = "saved-model-epoch_{epoch:02d}-acc_{val_accuracy:.3f}-dice_{val_dice_coef:.3f}-iou_{val_IoU:.3f}.hdf5"

        mck1 = os.path.join(save_path, f"{checkpoint_name}.hdf5")
        # save best val loss checkpoint
        mck2 = os.path.join(logs_path, filepath)
        # save each val loss checkpoint
        mck3 = os.path.join(save_path, f"best-valid-auc_{model_n}-{date_name}.hdf5")
        # save best val auc checkpoint

        callback_list = [
            ModelCheckpoint(mck1, monitor='val_IoU', mode='max', verbose=1, save_best_only=True),
            ModelCheckpoint(mck2, monitor='val_loss', verbose=1, save_best_only=True),
            ModelCheckpoint(mck3, monitor='val_dice_coef', mode='max', verbose=1, save_best_only=True,),
            TensorBoard(log_dir=tlogdir, histogram_freq=1,embeddings_freq=0,embeddings_layer_names=None,),
            LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]
        return callback_list
        
    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)

# building loss function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return dice
    
def tversky(y_true, y_pred):
    smooth = 1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.3
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def DBCEL(targets, inputs, smooth=1e-6):
    dice_loss = 1 - dice_coef(targets, inputs)
    BCE =  binary_crossentropy(targets, inputs)
    DBCE = BCE + dice_loss
    return DBCE

def FTL(targets, inputs, alpha=0.3, beta=0.7, gamma=2, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1-targets) * inputs))
    FN = K.sum((targets * (1-inputs)))
            
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = K.pow((1 - Tversky), gamma)
    
    return FocalTversky

# loading model
def model_loder():
    model = Unet(config['BACKBONE'], encoder_weights=None, input_shape=(None, None, 1))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # optimizer = tfa.optimizers.Lookahead(opt)
    optimizer = 'adam'
    model.compile(optimizer=optimizer, 
                loss=FTL, 
                metrics=['accuracy'
                        ,dice_coef
                        ,sm.metrics.IOUScore(name='IoU')
                        ,tf.keras.metrics.AUC()
                        ,tf.keras.metrics.Recall()])
    return model

# training process
def trainer():
    # data augmenation function
    # The rotation function is temporarily unavailable
    # Because the author has made a major update to the kit, some kits are affected and the rotation-related functions cannot be used.
    aug_img = A.Compose([
        A.HorizontalFlip(p=0.5),
        # A.Rotate(limit=5, p=0.5),
        # A.ShiftScaleRotate(rotate_limit=5,p=0.3),
        ],)  

    # dataloaders
    if not os.path.exists(save_path+ f'/history_log/'):
        os.makedirs(save_path + f'/history_log/')
    X_train, Y_train, X_val, Y_val = dataloader()
    img_data_gen = ImageDataAugmentor(augment=aug_img, input_augment_mode='image', seed=123)
    mask_data_gen = ImageDataAugmentor(augment=aug_img, input_augment_mode='mask',seed=123,)

    image_data_valid = ImageDataAugmentor(augment=None, input_augment_mode='image',seed=123)
    mask_data_valid = ImageDataAugmentor(augment=None, input_augment_mode='mask',seed=123)

    # build data generator (data augmenation)
    X_train_datagen = img_data_gen.flow(X_train, batch_size=batch_size)
    Y_train_datagen = mask_data_gen.flow(Y_train, batch_size=batch_size)
    train_generator = (pair for pair in zip(X_train_datagen, Y_train_datagen))

    X_valid_datagen = image_data_valid.flow(X_val, batch_size=batch_size)
    Y_valid_datagen = mask_data_valid.flow(Y_val, batch_size=batch_size)
    valid_generator = (pair for pair in zip(X_valid_datagen, Y_valid_datagen))
    # print(len(X_train_datagen), len(X_valid_datagen))
    del X_train, Y_train, X_val, Y_val

    # model loading
    model = model_loder()
    snapshot = SnapshotCallbackBuilder(nb_epochs = epochs, nb_snapshots = 1,init_lr = lr)
    print('-'*30,'\nFitting model...\n','-'*30)
    history = model.fit_generator(train_generator, epochs=epochs, 
                        steps_per_epoch= (len(X_train_datagen)*2),
                        shuffle=True, validation_data=valid_generator,
                        validation_steps= (len(X_valid_datagen)), 
                        callbacks=snapshot.get_callbacks())
    with open(save_path + f'/history_log/histroy-fold_{date_name}', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# main .py
if __name__ == '__main__':
    # Model SETTINGS
    model_n = config['model_name']
    # INITIAL SETTINGS
    nii_size = config['nii_size']
    epochs = config['epochs']
    lr = config['lr']
    print(type(lr))
    batch_size = int(config['batch_size'])
    date_name = config['date_name']
    MRI_nii_folder_path = config['MRI_nii_folder_path']
    save_path =config['save_path']
    
    checkpoint_name = f"{model_n}-{nii_size}-epochs_{epochs}-lr_{lr}-batch_{batch_size}-{date_name}"
    tlogdir = os.path.join(config['save_path'], checkpoint_name)
    # load path-url.npy and load label.npy
    # train_path_30T = np.load(MRI_nii_folder_path + 'T3_image_mask_path_train.npy')
    # valid_path_30T = np.load(MRI_nii_folder_path + 'T3_image_mask_path_valid.npy')

    # creative weight and tensorboard path folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save checkpoint (best/each)
    logs_path = save_path+ f'logs_{date_name}_CK/'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    # start train
    trainer() # valid_data: 1.5T or 3.0T
