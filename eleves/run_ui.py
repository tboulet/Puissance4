import argparse

from player import HumanPlayer, RandomPlayer
from ai_player import *
from ui_game import UIGame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1', default='AI')
    parser.add_argument('--p2', default='Human')
    parser.add_argument('--p3', default='player 3')
    args = parser.parse_args()

    AI = AIPlayer()
    AI.name = args.p1
    human = HumanPlayer()
    human.name = args.p2
    
    player3 = AIPlayer()
    player3.name = args.p3

    game = UIGame(human, AI)
