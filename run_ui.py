import argparse
from heuristic_functions import heuristic_bad

from player import HumanPlayer, RandomPlayer
from ai_player import AIPlayer, AlphaBetaPlayer, MinimaxPlayer, RLPlayer
from game_connect4.ui_game import UIGame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1', default='player 1')
    parser.add_argument('--p2', default='player 2')
    args = parser.parse_args()

    player1 = AIPlayer(heuristic=heuristic_bad, max_depth=4)
    player2 = AlphaBetaPlayer(heuristic=heuristic_bad, max_depth=4)

    player1.name = args.p1
    player2.name = args.p2
    
    game = UIGame(player1, player2)
