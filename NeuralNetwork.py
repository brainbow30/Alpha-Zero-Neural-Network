import datetime
import time

from flask import Flask
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from Board import *

app = Flask(__name__)

sys.path.append('../..')
from utils import *

from OthelloNN import OthelloNN as onnet

import json

with open('config.json') as json_data_file:
    config = json.load(json_data_file)

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 94,
    'cuda': False,
    'num_channels': 512,
    'checkpoint': config["checkpoints"]
})


class NeuralNetwork():
    def __init__(self, board):
        self.nnet = onnet(board, args)
        self.board_x, self.board_y = board.getBoardSize()
        self.action_size = board.getActionSize()

    def train(self, examples, filepath):
        # todo remove pi, only train with board and result, then when predicting only return predicted result of board
        """
        examples: list of examples, each example is of form (board, policy value, outcome of game)
        """
        try:
            self.nnet.model = load_model(filepath)
        except:
            print("no checkpoint")
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        checkpoint = ModelCheckpoint(filepath, monitor='v_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=args.batch_size,
                            callbacks=callbacks_list, epochs=args.epochs)
        self.nnet.clear()

    def predict(self, board, size):
        """
        board: np array with board
        """
        try:
            self.nnet.model = load_model("checkpoints/weights.best" + str(size) + ".hdf5")
        except:
            print("no checkpoint")

        start = time.time()
        # preparing input
        board = read.board(board, size)
        board = board[np.newaxis, :]
        # run
        pi, v = self.nnet.model.predict(board)
        self.nnet.clear()
        print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time() - start))
        return pi[0], v[0]


@app.route('/train/<int:boardSize>')
def train(boardSize):
    try:

        boardProperties = Board(boardSize)
        nn = NeuralNetwork(boardProperties)
        intBoards = read.file("training" + str(boardProperties.getN()) + ".txt", boardSize)
        args["batch_size"] = len(intBoards)
        filepath = "checkpoints/weights.best" + str(boardProperties.getN()) + ".hdf5"
        nn.train(intBoards, filepath)
        K.clear_session()
        return str(datetime.datetime.now()) + " Trained"
    except Exception as e:
        print(e)
        return "error"


@app.route('/predict/<int:boardSize>/<string:board>')
def predict(boardSize, board):
    try:
        K.clear_session()
        boardProperties = Board(boardSize)
        nn = NeuralNetwork(boardProperties)
        pi, v = nn.predict(board, boardSize)
        K.clear_session()
        return str(v[0])
    except Exception as e:
        print(e)
        return "error"


if __name__ == '__main__':
    app.run()
