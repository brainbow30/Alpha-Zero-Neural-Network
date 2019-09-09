import datetime
import time

from flask import Flask

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

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, policy value, outcome of game)
        """
        self.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=args.batch_size, epochs=args.epochs)
        self.save_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
        self.nnet.clear()

    def predict(self, board):
        """
        board: np array with board
        """
        self.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
        start = time.time()
        # preparing input
        board = read.board(board)
        board = board[np.newaxis, :]
        # run
        pi, v = self.nnet.model.predict(board)
        self.nnet.clear()
        print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time() - start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")

        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)


@app.route('/train/<int:boardSize>')
def train(boardSize):
    boardProperties = Board(boardSize)
    nn = NeuralNetwork(boardProperties)
    intBoards = read.file("training" + str(boardProperties.getN()) + ".txt")
    args["batch_size"] = len(intBoards)
    nn.train(intBoards)
    return str(datetime.datetime.now()) + " Trained"


@app.route('/predict/<int:boardSize>/<string:board>')
def predict(boardSize, board):
    boardProperties = Board(boardSize)
    nn = NeuralNetwork(boardProperties)
    pi, v = nn.predict(board)
    return str(v[0])


if __name__ == '__main__':
    app.run()
