import datetime
import sys
import time

import tensorflow as tf
from flask import Flask
from flask import request
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
modelLocation = "checkpoints/smallNNweights.best"
model = load_model(modelLocation + str(config["boardSize"]) + ".h5")

global testgraph
testgraph = tf.get_default_graph()
testModelLocation = "checkpoints/bigNNweights.best"
testmodel = load_model(testModelLocation + str(config["boardSize"]) + ".h5")

@app.route('/train/<int:boardSize>', methods=["PUT"])
def train(boardSize):
    global model
    global graph
    try:
        intBoards = read.trainingData(request.json["data"], boardSize)
        args["batch_size"] = len(intBoards)
        filepath = modelLocation + str(config["boardSize"]) + ".h5"
        input_boards, target_pis, target_vs = list(zip(*intBoards))
        target_pis = np.asarray(target_pis)
        input_boards = np.asarray(input_boards)
        target_vs = np.asarray(target_vs)
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        with graph.as_default():
            model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=args.batch_size, callbacks=callbacks_list,
                      epochs=args.epochs)
        K.clear_session()
        tf.reset_default_graph()
        graph = tf.get_default_graph()
        model = load_model(modelLocation + str(config["boardSize"]) + ".h5")
        time.sleep(10)
        return str(datetime.datetime.now()) + " Trained"
    except Exception as e:
        K.clear_session()
        tf.reset_default_graph()
        graph = tf.get_default_graph()
        model = load_model(modelLocation + str(config["boardSize"]) + ".h5")
        time.sleep(10)
        print(e)
        return "error"


@app.route('/predict/<int:size>/<string:board>')
def predict(size, board):
    try:
        board = read.board(board, size)
        board = board[np.newaxis, :]
        with graph.as_default():
            pi, v = model.predict(board)
        policyString = ""
        for i in pi[0]:
            policyString += str(i) + ","
        policyString = policyString[0:len(policyString) - 1]
        return policyString + ":" + str(v[0][0])
    except Exception as e:
        print(e)
        return "error"


@app.route('/testpredict/<int:size>/<string:board>')
def testpredict(size, board):
    try:
        board = read.board(board, size)
        board = board[np.newaxis, :]
        with testgraph.as_default():
            pi, v = testmodel.predict(board)
        policyString = ""
        for i in pi[0]:
            policyString += str(i) + ","
        policyString = policyString[0:len(policyString) - 1]
        return policyString + ":" + str(v[0][0])
    except Exception as e:
        print(e)
        return "error"


if __name__ == '__main__':
    app.run()
