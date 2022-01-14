import copy
import random
import os
import sys

import utils
from board import Board


class Game(object):
    """Generic class to run a game."""
    def __init__(
            self, player1, player2, cols=7, rows=6,
            verbose=True):
        self.board = Board(num_rows=rows, num_cols=cols)
        self.players = [player1, player2]
        self.verbose = verbose
        self.max_moves = 2 * self.board.num_rows * self.board.num_cols
        self.reset()

    def isOver(self):
        return self.winner is not None or self.board.isFull() \
            or self.moves > self.max_moves

    def reset(self, randomStart=False):
        self.board.reset()
        self.winner = None
        # Make sure one player is 1 the other -1
        self.players[0].color = 1
        self.players[1].color = -1
        self.currPlayer = int(random.random() > 0.5) if randomStart else 0
        self.moves = 0

    def mayShowDebug(self):
        if not self.verbose:
            return

        print(self.board, '\n')
        if not self.isOver():
            return

        if self.winner is not None:
            print("{0} ({1}) wins!".format(
                self.winner.name, Board.valueToStr(self.winner.color)))
        else:
            print("It's a draw!")

    @utils.timeout(5)
    def getColumn(self, player):
        #sys.stdout = open(os.devnull, 'w')  # disables print
        return player.getColumn(copy.deepcopy(self.board))

    def run(self, randomStart=False):
        """This method runs the game, alternating between the players."""
        self.reset(randomStart)
        while not self.isOver():
            player = self.players[self.currPlayer]
            try:
                col = self.getColumn(player)
            except Exception as e:
                if self.verbose:
                    print(e)
                print("Player %i has made a bad choice"%player)
                self.winner = -player
                break

            row = self.board.play(player.color, col)
            pos = (col, row)
            if pos not in self.board:
                continue

            self.mayShowDebug()
            self.winner = self.board.winner
            self.currPlayer = (self.currPlayer + 1) % 2

        self.mayShowDebug()
