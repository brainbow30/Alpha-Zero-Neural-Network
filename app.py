import datetime
import os
import sys

sys.path.append('../../')
import torch
import torch.optim as optim
import tensorflow as tf
from ResNet_PyTorch import NNet

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from flask import Flask
from flask import request
from flask_caching import Cache
import json
from matplotlib import pyplot as plt
import random
from utils import *

app = Flask(__name__)
config = {
    "DEBUG": True,  # some Flask specific configs
    "CACHE_TYPE": "simple",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
app.config.from_mapping(config)
cache = Cache(app)

with open('C:\\Users\\brain\\PycharmProjects\\Alpha-Zero-Neural-Network\\config.json') as json_data_file:
    config = json.load(json_data_file)

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': config["epochs"],
    'batch_size': 50,
    'cuda': config["cuda"],
    'num_channels': 512
})

previous_games = 0
previous_examples = []


@app.route('/train/<int:boardSize>', methods=["PUT"])
def train(boardSize):
    cache.delete_memoized(predict)
    cache.delete_memoized(testpredict)
    global model
    global previous_examples, previous_games
    examples = read.trainingData(request.json["data"], boardSize)
    previous_examples += examples
    previous_games += 1
    if (previous_games >= config["gamesBeforeTraining"]):
        previous_games = 0
        try:
            optimizer = optim.Adam(model.parameters())
            for epoch in range(args.epochs):
                print()
                print('EPOCH ::: ' + str(epoch + 1))
                model.train()
                pi_losses = AverageMeter()
                v_losses = AverageMeter()
                batch_idx = 0
                while batch_idx < int(len(previous_examples) / args.batch_size):
                    sample_ids = np.random.randint(len(previous_examples), size=args.batch_size)
                    boards, pis, vs = list(zip(*[previous_examples[i] for i in sample_ids]))
                    target_boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                    target_pis = torch.FloatTensor(np.array(pis))
                    target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                    # predict
                    if args.cuda:
                        target_boards, target_pis, target_vs = target_boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                    # measure data loading time

                    # compute output
                    out_pi, out_v = model(target_boards)
                    l_pi = loss_pi(target_pis, out_pi)
                    l_v = loss_v(target_vs, out_v)
                    total_loss = l_pi + l_v

                    # record loss
                    pi_losses.update(l_pi.item(), target_boards.size(0))
                    v_losses.update(l_v.item(), target_boards.size(0))

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    batch_idx += 1
                print("pi: ", pi_losses.val)
                print("v: ", v_losses.val)

            save_checkpoint(model, config["modelFolder"] + config["game"],
                            config["modelFile"] + "." + str(config["boardSize"]))
            previous_examples = []
            return str(datetime.datetime.now()) + " Trained"
        except Exception as e:
            print(e)
            model = load_checkpoint(model, config["modelFolder"] + config["game"],
                                    config["modelFile"] + "." + str(config["boardSize"]))
            if (args.cuda):
                model.cuda()
            return "error"
    else:
        print("examples added")
        print("previous games: ", previous_games)
        return "examples added"


@app.route('/predict/<int:size>/<string:board>')
@cache.memoize(120)
def predict(size, board):
    try:
        board = read.board(board, size)
        board = board[np.newaxis, :]
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(1, size, size)
        model.eval()
        with torch.no_grad():
            pi, v = model(board)
        pi = pi.data.cpu().numpy()
        v = v.data.cpu().numpy()
        d = np.random.dirichlet(np.ones(len(pi[0])))
        pi[0] = 0.75 * pi + 0.25 * d
        policyString = ""
        for i in pi[0]:
            policyString += str(i) + ","
        policyString = policyString[0:len(policyString) - 1]
        return policyString + ":" + str(v[0][0])
    except Exception as e:
        print(e)
        return "error"


@app.route('/testpredict/<int:size>/<string:board>')
@cache.memoize(120)
def testpredict(size, board):
    try:
        board = read.board(board, size)
        board = board[np.newaxis, :]
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(1, size, size)

        testmodel.eval()

        with torch.no_grad():
            pi, v = testmodel(board)

        pi = pi.data.cpu().numpy()
        v = v.data.cpu().numpy()
        d = np.random.dirichlet(np.ones(len(pi[0])))
        pi[0] = 0.75 * pi + 0.25 * d
        policyString = ""
        for i in pi[0]:
            policyString += str(i) + ","
        policyString = policyString[0:len(policyString) - 1]
        return policyString + ":" + str(v[0][0])
    except Exception as e:
        print(e)
        return "error"


@app.route('/plot/<string:results>/<string:evaluator>')
def plotResults(results, evaluator):
    results = results.split(";")
    results = results[:-1]
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = ["^", (8, 2, 0), "o", "+"]
    lss = ["--", "--", "-", "-"]
    cs = ["b", "g", "k", "r"]
    i = 0
    legend = True
    for parameter1Results in results:
        parameter1, tempResults = parameter1Results.split("#")
        tempResults = tempResults.split(",")
        parameter2s = []
        winRatios = []
        for result in tempResults:
            parameter2, winRatio = result.split(":")
            parameter2s.append(float(parameter2))
            winRatios.append(float(winRatio))
        if (parameter1 == "null"):
            ax.plot(parameter2s, winRatios)
            legend = False
        else:
            ax.plot(parameter2s, winRatios, c=cs[i], marker=markers[i], ls=lss[i], label=str(parameter1))
        i += 1

    plt.title(config["game"] + " " + evaluator + ' Evaluation')
    plt.ylabel('Win Ratio')
    plt.xlabel(evaluator)
    if (legend):
        plt.legend(loc=2)
    plt.savefig("graphs/" + evaluator + str(random.randint(1, 1000)))
    return "Graph Saved"


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]


def loss_v(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


def save_checkpoint(model, folder='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
        print("Checkpoint Directory does not exist! Making directory {}".format(folder))
        os.mkdir(folder)
    else:
        print("Checkpoint Directory exists! ")
    torch.save({
        'state_dict': model.state_dict(),
    }, filepath)


def load_checkpoint(model, folder='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise ("No model in path {}".format(filepath))
    map_location = None if args.cuda else 'cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    return model


model = NNet()
testmodel = NNet()
try:
    print("loading ", config["modelFile"])
    model = load_checkpoint(model, config["modelFolder"] + config["game"],
                            config["modelFile"] + "." + str(config["boardSize"]))
    print("loaded")
    print("loading ", config["testModelFile"])
    testmodel = load_checkpoint(testmodel, config["modelFolder"] + config["game"],
                                config["testModelFile"] + "." + str(config["boardSize"]))
    print("loaded ")
except Exception as e:
    print("loading failed")
    print(e)
if (args.cuda):
    model.cuda()
    testmodel.cuda()

if __name__ == '__main__':
    app.run()
