import sys

import tensorflow as tf
from keras.initializers import glorot_uniform

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import json
from utils import *

with open('C:\\Users\\brain\\PycharmProjects\\Alpha-Zero-Neural-Network\\config.json') as json_data_file:
    config = json.load(json_data_file)

sys.path.append('..')


def identity_block(X, f, filters, stage, block):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + "2a",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + "2b",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + "2c",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Add()([X, X_shortcut])
    X = LeakyReLU(alpha=0.01)(X)
    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + "2a",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + "2b",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = LeakyReLU(alpha=0.01)(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + "2c",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    X = LeakyReLU(alpha=0.01)(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + "1",
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)

    X = Add()([X, X_shortcut])
    X = LeakyReLU(alpha=0.01)(X)
    return X


def ResNet(boardSize, args):
    # Define the input as a tensor with shape input_shape
    board_x, board_y = (boardSize, boardSize)
    action_size = boardSize * boardSize
    input_boards = Input(shape=(board_x, board_y))
    X_input = Reshape((board_x, board_y, 1))(input_boards)
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='d')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    pi = Dense(action_size, activation='linear', name='pi', kernel_initializer=glorot_uniform(seed=0))(
        X)  # batch_size x self.action_size
    v = Dense(1, activation='tanh', name='v', kernel_initializer=glorot_uniform(seed=0))(X)  # batch_size x 1
    # Create model
    model = Model(inputs=input_boards, outputs=[pi, v], name="ResNet")
    model.compile(loss=['mean_squared_error', 'mean_squared_error'], optimizer=Adam(args.lr))

    return model


def save(model, boardSize):
    filename = input("Enter File Name:")
    model.save(config["modelFolder"] + config["game"] + "/" + str(filename) + "." + str(boardSize) + ".h5")
    print("Saved model to disk")


args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 100,
    'batch_size': 94,
    'cuda': True,
    'num_channels': 512,

})

boardSize = config["boardSize"]
model = ResNet(boardSize, args)
save(model, boardSize)
