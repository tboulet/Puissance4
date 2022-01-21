import time
from player import Player
from copy import copy, deepcopy
import sys
from utils import board2obs


class MinimaxPlayer(Player):
    """This player explore to a curtain depth the game tree using the minimax algorithm"""

    def __init__(self, heuristic, max_depth = 4):
        super().__init__()
        self.name = "minimax"
        self.max_depth = max_depth
        self.heuristic = heuristic

    def getColumn(self, board):
        return self.best_move(board)
    
    def best_move(self, board):
        '''
        Return the move that maximize the minimax evaluation of the score obtainable through this move.
        '''
        columns = board.getPossibleColumns()
        best_col = None
        best_value = -sys.maxsize
        
        for col in columns:
            future_board = deepcopy(board)
            future_board.play(player = self.color, col = col)
            value = self.minimax(future_board, maximize = False, depth = 0)
            if value > best_value:
                best_value = value
                best_col = col
        return best_col 
    
    def minimax(self, board, maximize, depth):
        '''
        Return the best (or worse depending on maximize) score among every possible scores for player self.color, 
        assuming we know well the results thanks to high depth or good heuristic function and that both player will play following minimax algo.
        '''
        color = self.color
        if board.winner == color:
            return 10000
        elif board.winner == -color:
            return -10000
        elif board.isFull():
            return 0
        if depth >= self.max_depth:
            return self.heuristic(board, color = color) #The heuristic evaluates the score of a board for Max, without exloring the tree.
        else:
            
            columns = board.getPossibleColumns()
            scores = list()
            for col in columns:
                future_board = deepcopy(board)
                future_board.play(player = color if maximize else -color, col = col)
                value = self.minimax(future_board, maximize = not maximize, depth = depth + 1)
                scores.append(value)
            
            return max(scores) if maximize else min(scores) 



class AlphaBetaPlayer(Player):
    """This player explore to a curtain depth the game tree using the minimax algorithm"""

    def __init__(self, heuristic, max_depth = 4):
        super().__init__()
        self.name = "alphabeta"
        self.max_depth = max_depth
        self.heuristic = heuristic
        
    def getColumn(self, board):
        return self.best_move(board)

    def best_move(self, board):
        alpha = - sys.maxsize
        beta = sys.maxsize
        columns = board.getPossibleColumns()
        best_col = None
        
        for col in columns:
            future_board = deepcopy(board)
            future_board.play(player = self.color, col = col)
            value = self.alphabeta(future_board, maximize = False, depth = 1, alpha=alpha, beta=beta)
            if value > alpha:
                alpha = value
                best_col = col
        return best_col 
        
    def alphabeta(self, board, maximize, depth, alpha, beta):
        
        color = self.color
        if board.winner == color:
            return 10000
        elif board.winner == -color:
            return -10000
        elif board.isFull():
            return 0
        if depth >= self.max_depth:
            return self.heuristic(board, color = color)
            
        columns = board.getPossibleColumns()
        if maximize:
            value_max = -sys.maxsize
            for col in columns:
                future_board = deepcopy(board)
                future_board.play(player = color, col = col)
                value = self.alphabeta(future_board, 
                                        maximize = False, 
                                        depth = depth + 1, 
                                        alpha=alpha, 
                                        beta=beta)
                value_max = max(value_max, value)
                if value_max >= beta:
                    return value_max
                alpha = max(alpha, value)
            return value_max
        
        else:
            value_min = sys.maxsize
            for col in columns:
                future_board = deepcopy(board)
                future_board.play(player = -color, col = col)
                value = self.alphabeta(future_board, 
                                        maximize = True, 
                                        depth = depth + 1, 
                                        alpha=alpha, 
                                        beta=beta)
                value_min = min(value_min, value)
                if value_min <= alpha:
                    return value_min
                beta = min(beta, value)
            return value_min
            
                    
class AIPlayer(Player):

    def __init__(self, heuristic, max_depth = 2):
        super().__init__()
        self.name = "AI_Player"
        self.heuristic = heuristic
        self.max_depth = max_depth
    
    def getColumn(self, board):
        '''
        This function is basically the policy pi : state --> action  or  pi : board --> column  
        '''
        return self.alphabeta(board)

    
    def alphabeta(self, board):

        def maxvalue(board, alpha, beta, depth):
            #terminal test
            if depth>=self.max_depth or board.winner!=None or board.isFull():
                return self.heuristic(board, self.color)
            v = -sys.maxsize
            for action in board.getPossibleColumns():
                boardcopy = deepcopy(board)
                boardcopy.play(self.color, action)
                v = max(v, minvalue(boardcopy, alpha, beta, depth+1))
                if v>=beta:
                    return v
                alpha = max(alpha,v)
            return v

        def minvalue(board, alpha, beta, depth):
            if depth>=self.max_depth or board.winner!=None or board.isFull():
                return self.heuristic(board, self.color)
            v = sys.maxsize
            for action in board.getPossibleColumns():
                boardcopy = deepcopy(board)
                boardcopy.play(-self.color, action)
                v = min(v, maxvalue(boardcopy, alpha, beta, depth+1))
                if v<=alpha:
                    return v
                beta = min(beta,v)
            return v

        meilleur_score = -sys.maxsize
        beta = sys.maxsize
        alpha = -sys.maxsize
        coup = None
        possibleplays = board.getPossibleColumns()
        if len(possibleplays)==0:
            print("Il ne devrait pas y avoir 0 coup à jouer")
            raise
        elif len(possibleplays)==1:
            return possibleplays[0]
        else:
            for action in possibleplays:
                boardcopy = deepcopy(board)
                boardcopy.play(self.color, action)
                v = minvalue(boardcopy, alpha, beta, 1)
                if v>meilleur_score:
                    meilleur_score = v
                    coup = action
            return coup
 
 
 
 
class RLPlayer(Player):

    """Player class corresponding to a RL agent """

    def __init__(self):
        super().__init__()
        self.name = "AI_Player"

    
    def getColumn(self, board):
        '''
        This function is basically the policy pi : state --> action  or  pi : board --> column  
        '''
        return self.alphabeta(board)