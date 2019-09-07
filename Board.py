import sys

sys.path.append('..')


class Board():

    def __init__(self, n):
        self.n = n

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n * self.n + 1
