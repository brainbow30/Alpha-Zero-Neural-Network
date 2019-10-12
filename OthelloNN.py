import sys

from keras.layers import *
from keras.models import *
from keras.optimizers import *

sys.path.append('..')

class OthelloNN():
    def __init__(self, boardSize, args):
        # game params
        self.board_x, self.board_y = (boardSize, boardSize)
        self.action_size = boardSize * boardSize
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_x, self.board_y))  # s: batch_size x board_x x board_y
        x_image = Reshape((self.board_x, self.board_y, 1))(self.input_boards)  # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='same', use_bias=False)(
                x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='same', use_bias=False)(
                h_conv1)))  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(
                h_conv2)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(
                h_conv3)))  # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(
            BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))  # batch_size x 1024
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1
        self.model = Model(inputs=self.input_boards, outputs=self.v)
        self.model.compile(loss='mean_squared_error', optimizer=Adam(args.lr))

    def clear(self):
        K.clear_session()
