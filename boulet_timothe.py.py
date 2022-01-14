from player import Player
from copy import deepcopy
import sys


# Que devez-vous faire avec ce fichier ?
# 1) Changez le nom de l'IA: ligne 18
# 2) Ecrire la fonction heuristique qui prend en entrée un board et qui retourne un entier
# 3) Surtout: ne touchez à rien d'autre, ne renommez rien à l'intérieur de ce fichier ou les autres
# 4) Changez le nom du fichier avant de nous le rendre: NOM_prenom.py


class AIPlayer(Player):
    """This player should implement a heuristic along with a min-max and alpha
    beta search to """

    def __init__(self):
        super().__init__()
        self.name = "IA_Timothé_Boulet" #mettez ici le nom de votre IA : ne laissez pas ce nom svp

    
    def getColumn(self, board):
        return self.alphabeta(board)

    
#vous pouvez changer le chiffre 4 qui indique la profondeur à laquelle votre
#IA va aller dans l'arbre de jeu.
    def alphabeta(self, board, profondeurmax=4):


        def heuristic(board,color):
            """
                Evalue un plateau de jeu en retournant un entier.
                board: le plateau de jeu
                color: le joueur pour qui on calcule le meilleur coup (Max)
                        soit -1, soit +1
            """
            #print("Choix du coup pour la grille: ", board)
            # TODO(student): implement this!
            # il est attendu que la fonction renvoie un entier
            # une grande valeur étant plutôt favorable pour Max
            if board.winner == color:
                return 100000 #mettez la valeur max de votre heuristique ici
            elif board.winner == -color:
                return -100000 #mettez la valeur min de votre heuristique ici
            elif board.isFull():
                #pas de gagnant, mais la grille est pleine (match nul)
                 return 0

            else:
                #On considère qu'un tableau est bon pour Max si il y a beaucoup de suites de jetons alignés qui peuvent, si on arrivait à rajouter un voir deux jetons, faire gagner la partie. 

                n_seq2 = 0
                n_seq3 = 0
                
                #On crée un liste dont chaqué élément est une liste modélisant une ligne/colonne/diagonale
                ListeDroites = []
                #Recherche dans les lignes
                for row in range(6):
                    Row = board.getRow(row)
                    ListeDroites.append(Row)
                #Reherche dans les colonnes
                for col in range(7):
                    Col = board.getCol(col)
                    ListeDroites.append(Col)
                #Recherche dans les diagonales
                for up in [True, False]:
                    for col in range(7):
                        Diag = board.getDiagonal(up, col, 0)
                        ListeDroites.append(Diag)
                         
                #On cherche les séquences de 4 chiffres contenant trois 1 et un seul 0 (pour Max)
                #On cherche en seconde priorité les séquences de 3 chiffres contenant deux 1 et un 0.
                for color_ in [-1,1]:
                    for Droite in ListeDroites:
                        long = len(Droite)
                        if long>=4:
                            for i in range(long-3):
                                sequence = Droite[i:i+4]
                                if sequence.count(color_)==3 and 0 in sequence:
                                    n_seq3+=color_
                                elif sequence.count(color_)==2 and sequence.count(0)==2:
                                    n_seq2+=color_
                return (10*n_seq3+n_seq2)*color
                

#Ne pas toucher les fonctions qui suivent svp        
        def maxvalue(board, alpha, beta, hauteur):
            #terminal test
            if hauteur>=profondeurmax or board.winner!=None:
                return heuristic(board, self.color)
            v = -sys.maxsize
            for action in board.getPossibleColumns():
                boardcopy = deepcopy(board)
                boardcopy.play(self.color, action)
                v = max(v, minvalue(boardcopy, alpha, beta, hauteur+1))
                if v>=beta:
                    return v
                alpha = max(alpha,v)
            return v

        def minvalue(board, alpha, beta, hauteur):
            if hauteur>=profondeurmax or board.winner!=None:
                return heuristic(board, self.color)
            v = sys.maxsize
            for action in board.getPossibleColumns():
                boardcopy = deepcopy(board)
                boardcopy.play(-self.color, action)
                v = min(v, maxvalue(boardcopy, alpha, beta, hauteur+1))
                if v<=alpha:
                    return v
                beta = min(beta,v)
            return v

        meilleur_score = -sys.maxsize
        beta = sys.maxsize
        coup = None
        for action in board.getPossibleColumns():
            boardcopy = deepcopy(board)
            boardcopy.play(self.color, action)
            v = minvalue(boardcopy, meilleur_score, beta, 1)
            if v>meilleur_score:
                meilleur_score = v
                coup = action

        return coup
            
        
            

