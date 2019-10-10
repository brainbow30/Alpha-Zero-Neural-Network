import json
import os

import numpy as np

with open('config.json') as json_data_file:
    config = json.load(json_data_file)

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class read():
    def file(filename, size):
        boardTuples = []
        filepath = os.path.join(config["intBoards"], filename)
        text_file = open(filepath, "r")
        lines = text_file.readlines()
        for row in lines:
            try:
                state = str.split(row, "\n")
                state = state[0]
                board, result = str.split(state, ":")
                npboard = read.board(board, size)
                boardTuples.append((npboard, result))
            except:
                print("Error")


        text_file.close()
        return boardTuples

    def board(board, size):
        board = str.split(board, ",")
        board = np.array(board)
        tempboard = []
        for i in range(0, size):
            row = []
            for j in range(0, size):
                row.append(board[(i * size) + j])
            tempboard.append(row)
        return np.array(tempboard)
