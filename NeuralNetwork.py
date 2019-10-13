import datetime
import sys

import tensorflow as tf
from flask import Flask
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from utils import *

app = Flask(__name__)

sys.path.append('../..')


with open('config.json') as json_data_file:
    config = json.load(json_data_file)

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 100,
    'batch_size': 94,
    'cuda': False,
    'num_channels': 512,
    'checkpoint': config["checkpoints"]
})

global graph
graph = tf.get_default_graph()
# todo make model an array of all the models of different sizes
model = load_model("checkpoints/weights.best" + str(config["boardSize"]) + ".h5")

@app.route('/train/<int:boardSize>')
def train(boardSize):
    global model
    try:
        intBoards = read.file("training" + str(boardSize) + ".txt", boardSize)
        args["batch_size"] = len(intBoards)
        filepath = "checkpoints/weights.best" + str(boardSize) + ".h5"
        input_boards, target_vs = list(zip(*intBoards))
        input_boards = np.asarray(input_boards)
        target_vs = np.asarray(target_vs)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        with graph.as_default():
            model.fit(x=input_boards, y=target_vs, batch_size=args.batch_size, callbacks=callbacks_list,
                      epochs=args.epochs)

        return str(datetime.datetime.now()) + " Trained"
    except Exception as e:
        print(e)
        return "error"
    K.clear_session()
    model = load_model("checkpoints/weights.best" + str(config["boardSize"]) + ".h5")

@app.route('/predict/<int:size>/<string:board>')
def predict(size, board):
    try:
        board = read.board(board, size)
        board = board[np.newaxis, :]
        with graph.as_default():
            v = str(model.predict(board)[0][0])
        return v
    except Exception as e:
        print(e)
        return "error"


if __name__ == '__main__':
    app.run()
