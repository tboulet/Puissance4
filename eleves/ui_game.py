import logging
import tkinter
import multiprocessing

from board import Board
from game import Game


class UIGame(Game):
    """Play the connect 4 game, but in a Tkinter GUI"""
    def __init__(self, player1, player2, dbg=None):
        super().__init__(player1, player2, verbose=True)
        self.tk = tkinter.Tk()
        self.width = 600
        self.height = 500
        self.timeout = 3 * 1000  # not the limitation for the AI...
        self.cellH = self.height / self.board.num_rows
        self.cellW = self.width / self.board.num_cols

        self.labels = []
        for k, player in enumerate(self.players):
            self.labels.append(tkinter.Label(
                self.tk, text=player.name, fg=self.getColor(player.color),
                font=("Helvetica", 16)))
            self.labels[-1].grid(row=0, column=k)

        self.info = tkinter.StringVar()
        self.infoLabel = tkinter.Label(
            self.tk, textvariable=self.info, font=("Helvetica", 16))
        self.infoLabel.grid(row=1, column=0, columnspan=2)

        if dbg:
            self.infoLabel = tkinter.Label(
                self.tk, text=dbg, font=("Helvetica", 14))
            self.infoLabel.grid(row=0, column=2, rowspan=3, sticky='n')

        self.canvas = tkinter.Canvas(
            height=self.height, width=self.width)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.grid(row=2, column=0, columnspan=2)

        self.reset()

        # A bit to make sure to stop when it's over.
        self.over = False
        # Start the game if the player to start is not Human
        self.run()

        self.tk.mainloop()

    @staticmethod
    def getColor(value):
        return 'red' if value == 1 else 'blue'

    def renderOne(self, i, j):
        x = self.cellW * i
        y = self.height - self.cellH * j
        color = self.getColor(self.board[i, j])
        self.canvas.create_oval(
            x, y, x + self.cellW, y - self.cellH, fill=color)

    def render(self, board):
        for i in range(board.num_cols):
            for j in range(board.num_rows):
                if board[i, j] != 0:
                    self.renderOne(i, j)

    def run(self):
        if self.over:
            return

        player = self.players[self.currPlayer]
        if player.HUMAN:
            return

        self.text = ''
        try:
            col = self.getColumn(player)
        except multiprocessing.context.TimeoutError as e:
            reason = "{} took too long".format(player.name)
            logging.error(reason)
            col = -1
            self.text += reason
            self.info.set(self.text)
        except Exception as e:
            if self.verbose:
                raise e
            reason = "{} throw an exception !".format(player.name)
            logging.error(reason)
            self.text = reason
            self.info.set(self.text)
            return self.mayMakeCurrentPlayerLoose()

        self.play(col)

    def click(self, event):
        """Reponse to a click event, for human player only."""
        player = self.players[self.currPlayer]
        if player.HUMAN:
            col = int(event.x / self.cellW)
            self.play(col)

    def play(self, col):
        """The current player puts a token on the column given as input"""
        if self.over:
            return

        self.moves += 1

        player = self.players[self.currPlayer]
        row = self.board.play(player.color, col)
        pos = (col, row)

        # AI mistake ? not good. Skip your turn
        if pos in self.board:
            self.renderOne(col, row)

        self.currPlayer = (self.currPlayer + 1) % 2
        if self.board.winner is None:
            self.winner = None
        elif self.board.winner==-1:
            self.winner = self.players[1]
        else:
            self.winner = self.players[0]
        if not self.handleEnd():
            self.tk.after(20, self.run)

    def mayMakeCurrentPlayerLoose(self):
        player = self.players[self.currPlayer]
        if not player.HUMAN:
            self.winner = self.players[(self.currPlayer + 1) % 2]
            self.over = True
            self.handleEnd()

    def handleEnd(self):
        if not self.isOver() and not self.over:
            return False

        text = "It's a draw!"
        if self.winner is not None:
            text = "{0} ({1}) wins!".format(
                self.winner.name, Board.valueToStr(self.winner.color))
        self.text += '\n{}'.format(text)
        self.info.set(self.text)
        self.over = True
        self.tk.after(self.timeout, self.tk.destroy)
        return True
