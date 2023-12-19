import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

act = 'relu'
# act = keras.layers.LeakyReLU(alpha=0.3)
def get_models(depth=32, width=128, height=128, class_num=3, classification_layer='sigmoid'):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((depth, width, height, 1))

    cx1 = layers.Conv3D(filters=64, kernel_size=3)(inputs)
    ax1 = layers.Activation(act)(cx1)
    px1 = layers.MaxPool3D(pool_size=(2, 2, 2))(ax1)
    bx1 = layers.BatchNormalization()(px1)
    # print(bx1.shape)
    
    cx2 = layers.Conv3D(filters=64, kernel_size=3)(bx1)
    ax2 = layers.Activation(act)(cx2)
    px2 = layers.MaxPool3D(pool_size=(1, 2, 2))(ax2)
    bx2 = layers.BatchNormalization()(px2)
    # print(bx2.shape)
    
    cx4 = layers.Conv3D(filters=128, kernel_size=3)(bx2)
    ax4 = layers.Activation(act)(cx4)
    px4 = layers.MaxPool3D(pool_size=(2, 2, 2))(ax4)
    bx4 = layers.BatchNormalization()(px4)
    # print(bx4.shape)
    
    cx5 = layers.Conv3D(filters=256, kernel_size=3)(bx4)
    ax5 = layers.Activation(act)(cx5)
    px5 = layers.MaxPool3D(pool_size=(2, 2, 2))(ax5)
    bx5 = layers.BatchNormalization()(px5)
    # print(bx5.shape)

    x = layers.GlobalAveragePooling3D()(bx5)

    x = layers.Dense(units=512)(x)
    x = layers.Activation(act)(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(units=256)(x)
    x = layers.Activation(act)(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(units=class_num, activation=classification_layer, name='classification')(x)

    # Define the model.
    # model = keras.Model(inputs, [outputs, outputs2], name="3dcnn")
    model = keras.Model(inputs, outputs, name="3dcnn")
    
    return model
