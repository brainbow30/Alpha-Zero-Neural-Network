import os

import numpy as np


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class readBoards():
    def readfile(self, filename):
        boardTuples = []
        filepath = os.path.join("C:/Users/brain/IdeaProjects/Othello/intBoards/", filename)
        text_file = open(filepath, "r")
        lines = text_file.readlines()
        for row in lines:
            state = str.split(row, "\n")
            state = state[0]
            board, policy, result = str.split(state, ":")
            board = str.split(board, ",")
            board = np.array(board)
            tempboard = []
            for i in range(0, 6):
                row = []
                for j in range(0, 6):
                    row.append(board[(i * 6) + j])
                tempboard.append(row)
            npboard = np.array(tempboard)
            boardTuples.append((npboard, policy, result))

        text_file.close()
        return boardTuples
