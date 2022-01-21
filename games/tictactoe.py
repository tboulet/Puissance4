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

 
class TicTacToeBoard():
    
    def __init__(self, size):
        self.goal = size
        self.size = size
        self.board = list(list(0 for _ in range(size)) for _ in range(size))

    def flatten_board(self):
        return sum(L for L in self.board)
    
    def isFull(self):
        return not 0 in self.flatten_board()
    
    def reset(self):
        self.board = list(list(0 for _ in range(self.size)) for _ in range(self.size))
        
    def __repr__(self):
        string = ''
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:
                    string += '|X|'
                elif self.board[i][j] == -1:
                    string += '|O|'
                else:
                    string += '| |'
            string += '\n'
        return string
    
    def play(self, color, action):
        i, j = action // self.ize, asction % self.size
        if self.board[i][j] == - color:
            print(f"Warning : player {color} played on a box already taken by its opponent.")
            return
        self.board[i][j] = color
        self.winner = self._getWinner()
        
    def _getWinner(self):
        for color in [-1,1]:
            for line in self.board:
                if line.count(color) == self.goal:
                    return color
            for j in range(self.size):
                column = [self.board[i][j] for i in range(self.size)]
                if column.count(color) == self.goal:
                    return color
            diag1 = [self.board[i][i] for i in range(self.size)]
            if diag1.count(color) == self.goal:
                return color
            diag2 = [self.board[self.size-i][i] for i in range(self.size)]
            if diag2.count(color) == self.goal:
                return color
            
    def get_illegal_actions(self):
        board_f = self.flatten_board()
        return [i for i, box in enumerate(board_f) if box != 0]
    
    def get_possible_actions(self):
        board_f = self.flatten_board()
        return [i for i, box in enumerate(board_f) if box == 0]

    
class TicTacToeGame(object):
    """Generic class to run a game."""
    def __init__(
            self, player1, player2, size=3, goal=3,
            verbose=True):
        self.board = TicTacToeBoard(size=size, goal=goal)
        self.players = [player1, player2]
        self.verbose = verbose
        self.max_moves = 2 * size ** 2
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
                self.winner.name, self.winner.color))
        else:
            print("It's a draw!")

    # @utils.timeout(.5)
    def getAction(self, player):
        return player.getAction(copy.deepcopy(self.board))

    def run(self, randomStart=False):
        """This method runs the game, alternating between the players."""
        self.reset(randomStart)
        while not self.isOver():
            player = self.players[self.currPlayer]
            try:
                action = self.getAction(player)
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

            self.board.play(player.color, action)

            if self.verbose: self.mayShowDebug()
            if self.board.winner == None:
                self.winner = None
            elif self.board.winner == -1:
                self.winner = self.players[1]
            else:
                self.winner = self.players[0]
            
            self.currPlayer = (self.currPlayer + 1) % 2

        if self.verbose: self.mayShowDebug()