import sys

from keras.layers import *
from keras.models import *
from keras.optimizers import *

from utils import *

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
            Conv2D(args.num_channels, 4, padding='same', use_bias=False)(
                x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 4, padding='same', use_bias=False)(
                h_conv1)))  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(args.num_channels, 4, padding='valid', use_bias=False)(
                h_conv2)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv3_flat = Flatten()(h_conv3)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(
            BatchNormalization(axis=1)(Dense(1024, use_bias=False)(h_conv3_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(512, use_bias=False)(s_fc1))))  # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))


    def clear(self):
        K.clear_session()

    def save(self):
        filename = input("Enter File Name:")
        self.model.save(config["modelFolder"] + config["game"] + "/" + str(filename) + "." + str(self.board_x) + ".h5")
        print("Saved model to disk")


args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 100,
    'batch_size': 94,
    'cuda': False,
    'num_channels': 512,

})
# create new model
nn = OthelloNN(config["boardSize"], args)
nn.save()
