

def heuristic_bad(board,color):
            """
                Evaluate a board. The relative values of the board according to heuristic allows the agent to chose the optimal move.
                board : the board to evaluate
                color : the player for who the best action is computed. 1 or -1.
                Return : an evaluation of the board. 0 means the board is equivalent for both players.
            """
            if board.winner == color:
                return 10000
            elif board.winner == -color:
                return -10000
            elif board.isFull():
                return 0
            else:
                
                #Heuristic home-made
                n_seq2 = 0
                n_seq3 = 0
                
                seqs = []
                cols = board.num_cols
                rows = board.num_rows
                #Research in lines
                for row in range(rows):
                    Row = board.getRow(row)
                    seqs.append(Row)
                #Research in columns
                for col in range(cols):
                    Col = board.getCol(col)
                    seqs.append(Col)
                #Research in diagonals
                for up in [True, False]:
                    for col in range(cols):
                        Diag = board.getDiagonal(up, col, 0)
                        seqs.append(Diag)
                
                #We search for sequences of three 1 (or -1 depending of color) and one 0.
                #We search in second hand for sequences of two 1 and two 0.                         
                for color_ in [-1,1]:
                    for seq in seqs:
                        L = len(seq)
                        if L>=4:
                            for i in range(L-3):
                                sequence = seq[i:i+4]
                                if sequence.count(color_)==3 and 0 in sequence:
                                    n_seq3+=color_
                                elif sequence.count(color_)==2 and sequence.count(0)==2:
                                    n_seq2+=color_
                return (10*n_seq3+n_seq2)*color
