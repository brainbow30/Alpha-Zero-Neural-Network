import time

from Board import *

sys.path.append('../..')
from utils import *

from OthelloNN import OthelloNN as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 94,
    'cuda': False,
    'num_channels': 512,
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
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=args.batch_size, epochs=args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]


board = Board(6)
nn = NeuralNetwork(board)
readBoards = readBoards()
nn.train(readBoards.readfile("training.txt"))
