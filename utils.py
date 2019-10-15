import json

import numpy as np

with open('config.json') as json_data_file:
    config = json.load(json_data_file)

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class read():
    def trainingData(boards, size):
        boards = boards[2:len(boards) - 2]
        boards = boards.split("],[")
        boardTuples = []
        for row in boards:
            row = row.split("],")
            try:
                board = row[0][1:]
                result = row[1]
                npboard = read.board(board, size)
                boardTuples.append((npboard, result))
            except:
                print("Error")
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
