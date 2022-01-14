import abc
import logging
import random

from board import Board


class Player(metaclass=abc.ABCMeta):
    """This represents a player. By default a player who plays from the command
    line."""

    HUMAN = False

    def __init__(self):
        # In a game, the color should be 1 or -1
        self.color = 0
        self.name = ""

    @abc.abstractmethod
    def getColumn(self, board):
        pass

    def observe(self, board, winner):
        pass

    def save(self, basename):
        pass


class HumanPlayer(Player):
    """A human player from the command line or the GUI"""

    HUMAN = True

    def __init__(self):
        self.name = ""

    def getColumn(self, board):
        """By default, we play in command line, UI subclasses must overwrite
        this method."""
        colStr = input(
            "{0} ({1}): ".format(self.name, Board.valueToStr(self.color)))
        if colStr.isnumeric():
            return int(colStr)

        logging.error(
            "Column should be a value in 0-{0}".format(board.num_cols))
        return -1


class RandomPlayer(Player):
    """A player that randomly picks up a valid column"""

    def __init__(self):
        self.name=""
        
    def getColumn(self, board):
        columns = board.getPossibleColumns()
        if columns:
            return random.choice(columns)
