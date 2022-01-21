import utils

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
