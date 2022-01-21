from ai_player import AIPlayer, RLPlayer
from player import RandomPlayer, Player

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from utils import board2obs

import random
import copy
import sys
import os
import time

class Board(object):
    """This class represents the board of the connect4 games, with all the
    necessary operations to play a game"""
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.diagRanges = {
            True: range(-self.num_rows, self.num_cols),
            False: range(self.num_cols + self.num_rows)
        }
        self.winner = None
        self.reset()

    def reset(self):
        self.board = [[0] * self.num_rows for i in range(self.num_cols)]

    def __contains__(self, position):
        return 0 <= position[0] < self.num_cols and \
            0 <= position[1] < self.num_rows

    def __getitem__(self, position):
        if isinstance(position, tuple):
            if position in self:
                return self.board[position[0]][position[1]]
        else:
            return self.board[position]

    @staticmethod
    def valueToStr(value):
        toStr = {1: "x", -1: 'o'}
        return toStr.get(value, " ")

    def __repr__(self):
        rows = []
        for i in range(self.num_rows):
            values = self.getRow(i)
            rows.append(
                "|{0}|".format("|".join(map(self.valueToStr, values))))

        return "\n".join(reversed(rows))

    def play(self, player, col: int) -> int:
        """Player `player` puts a token at column `col`.
        Modifies the board and returns the row at which the token landed.
        """
        if col >= self.num_cols or col < 0:
            return -1

        row = self.getHeight(col)
        if row < self.num_rows:
            self.board[col][row] = player
        else:
            raise

        self.winner = self._getWinner((col,row))

        return row
    
    def get_illegal_columns(self):
        ill_cols = list(0 for _ in range(self.num_cols))
        for col in range(self.num_cols):
            if self.getHeight(col) >= self.num_rows:
                ill_cols[col] = 1
        return ill_cols
        
    def _getWinner(self, pos):
        """Returns the player (boolean) who won, or None if nobody won"""
        tests = []
        tests.append(self.getCol(pos[0]))
        tests.append(self.getRow(pos[1]))
        tests.append(self._getDiagonalIntern(True, pos[0] - pos[1]))
        tests.append(self._getDiagonalIntern(False, pos[0] + pos[1]))
        for test in tests:
            color, size = utils.longest(test)
            if size >= 4 and color!=0:
                return color

    def getHeight(self, col):
        """Returns the current height on the column `col`"""
        row = self.num_rows
        for i in range(self.num_rows):
            if self.board[col][i] == 0:
                row = i
                break
        return row

    def getPossibleColumns(self):
        """Returns all the possible columns that can be played"""
        result = []
        for col in range(self.num_cols):
            row = self.getHeight(col)
            if row < self.num_rows:
                result.append(col)

        return result

    def getRow(self, row):
        return list(map(lambda x: x[row], self.board))

    def getCol(self, col):
        return self.board[col]

    def _getDiagonalIntern(self, up, shift):
        """
         Down: x + y = shift
         Up: x - y = shift
        """
        result = []
        if up:
            for col in range(shift, self.num_cols):
                pos = (col, col - shift)
                if pos in self:
                    result.append(self[pos])
        else:
            for col in range(shift + 1):
                pos = (col, shift - col)
                if pos in self:
                    result.append(self[pos])
        return result

    def getDiagonal(self, up, col, li):
        if up:
            return self._getDiagonalIntern(up, col-li)
        else:
            return self._getDiagonalIntern(up, col+li)     

    def isFull(self):
        numEmpty = 0
        for column in self.board:
            for value in column:
                numEmpty += int(value == 0)

        return numEmpty == 0



class GameConnect4(object):
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

    def getColumn(self, player):
        return player.getColumn(copy.deepcopy(self.board))

    def run(self, randomStart=False):
        """This method runs the game, alternating between the players."""
        self.reset(randomStart)
        while not self.isOver():
            player = self.players[self.currPlayer]
            try:
                col = self.getColumn(player)
            except TimeoutError as te:
                print("Player %s has made a timeout"%player.name)
                self.winner = self.players[(self.currPlayer + 1) % 2]
                break
            except Exception as e:
                if self.verbose:
                    print(e)
                print("Player %s has made a bad choice"%player.name)
                self.winner = self.players[(self.currPlayer + 1) % 2]
                break

            row = self.board.play(player.color, col)
            pos = (col, row)
            if pos not in self.board:
                continue

            if self.verbose: self.mayShowDebug()
            if self.board.winner == None:
                self.winner = None
            elif self.board.winner == -1:
                self.winner = self.players[1]
            else:
                self.winner = self.players[0]
            
            self.currPlayer = (self.currPlayer + 1) % 2

        if self.verbose: self.mayShowDebug()
        
        
        
        
class EnvConnect4(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_cols = 7, num_rows = 6, opponent = RandomPlayer()):
        super().__init__()
        self.opponent = opponent
        self.game = GameConnect4(RandomPlayer(), self.opponent, cols = num_cols, rows= num_rows)
        
        self.verbose = True
        self.max_moves = 2 * self.game.board.num_rows * self.game.board.num_cols
        
        self.action_space = spaces.Discrete(self.game.board.num_cols)
        self.observation_space = spaces.Box(low = -1.0, high = 1.0, shape = (num_rows,num_cols))
        #self.reset(randomStart = False)
        


    def step(self, action):
                
        #RL Player plays
        row = self.game.board.play(1, col = action)
        pos = (action, row)
        if pos not in self.game.board:
            print("Pos not in self board")
            raise
        
        if self.verbose: self.game.mayShowDebug()
                
        #Opponent plays
        if not self.game.isOver():
            col_opponent = self.opponent.getColumn(self.game.board)
            row = self.game.board.play(-1, col = col_opponent)
            pos = (action, row)
            if pos not in self.game.board:
                print("Pos not in self board")
                raise
            
            if self.verbose: self.game.mayShowDebug()
            
        #Check who won  
        if self.game.board.winner == -1:
            reward = -1.0
            done = True
        elif self.game.board.winner == 1:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = self.game.isOver() 
        
        masked_actions = self.game.board.get_illegal_columns()
        info = {"mask" : masked_actions}
        
        next_obs = self.get_state()
        
        return next_obs, reward, done, info    
        
    def reset(self):
        self.game.reset(randomStart=False)
        board = self.game.board
        return self.get_state()
    
    def render(self, mode='human', close=False):
        pass
    
    def get_state(self):
        "Traduces a board (list of list) into a 2D array"
        return board2obs(self.game.board.board)
        