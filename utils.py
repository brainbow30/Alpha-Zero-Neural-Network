
import numpy as np


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class read():
    def trainingData(boards, size):

        boards = boards[3:len(boards) - 2]
        boards = boards.split("],[[")
        boardTuples = []
        for row in boards:
            row = row.split("],")
            try:
                board = row[0]
                pi = row[1][1:]
                result = row[2]
                npboard = read.board(board, size)
                nppi = read.pi(pi)

                boardTuples.append((npboard, nppi, result))
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

    def pi(pi):
        pi = str.split(pi, ",")
        tempboard = []
        for i in pi:
            tempboard.append(float(i))
        return np.array(tempboard)
