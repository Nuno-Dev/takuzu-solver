# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo al017:
# 99220 Francisco Lopes
# 99292 Nuno Martins

import sys
import numpy as np

from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    compare_searchers,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)

class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

class Board:
    """Representação interna de um tabuleiro de Takuzu."""
    def __init__(self, n):
        self.grid = [[2 for _ in range(n)] for _ in range(n)]
        self.n = n

    def size (self):
        return self.n

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.grid[row][col]    

    def set_number(self, row, col, value):
        self.grid[row][col] = value    

    def adjacent_vertical_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        a = self.grid[row-1][col] if row-1>=0 else None
        b = self.grid[row+1][col] if row+1<self.size() else None
        return (a,b)

    def adjacent_horizontal_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        a = self.grid[row][col-1] if col-1>=0 else None
        b = self.grid[row][col+1] if col+1<self.size() else None
        return (a,b)

    def is_valid(self):
        n = self.size()
        # equal 1s and 0s in rows
        for i in range(n):
            ones=zeros=twos=0
            for j in range(n):
                zeros += self.get_number(i,j) == 0
                ones += self.get_number(i,j) == 1
                twos += self.get_number(i,j) == 2
            if n%2==0:
                if ones>zeros+twos or zeros>ones+twos: return False
            else:
                if ones>zeros+twos+1 or zeros>ones+twos+1: return False

        # equal 1s and 0s in cols
        for j in range(n):
            ones=zeros=twos=0
            for i in range(n):
                zeros += self.get_number(i,j) == 0
                ones += self.get_number(i,j) == 1
                twos += self.get_number(i,j) == 2
            if n%2==0:
                if ones>zeros+twos or zeros>ones+twos: return False
            else:
                if ones>zeros+twos+1 or zeros>ones+twos+1: return False
        
        # max 2 adjacent equals
        for i in range(n):
            for j in range(n):
                cur = self.get_number(i,j)
                if cur == 2: continue
                a,b = self.adjacent_vertical_numbers(i,j)
                c,d = self.adjacent_horizontal_numbers(i,j)
                if cur==a==b or cur==c==d: return False
        
        # no repeated rows
        seen = set()
        for row in self.grid:
            if row.count(2)==0:
                if tuple(row) in seen: return False
                seen.add(tuple(row))
        
        # no repeated cols
        seen=set()
        for col in list(zip(*self.grid)):
            if col.count(2)==0:
                if tuple(col) in seen: return False
                seen.add(tuple(col))

        return True

    def print_board(self):
        arr = [[str(self.grid[i][j]) for j in range(self.n)] for i in range(self.n)]
        result = []
        for line in arr:
            result.append('\t'.join(line))
            result.append('\n')
        return ''.join(result)

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board. """
        n = int(sys.stdin.readline())
        board = Board(n)
        for i in range(n):
            row = sys.stdin.readline()
            board.grid[i] = list([int(el) for el in row.rstrip('\n').split('\t')])
        return board

    


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = TakuzuState(board)

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        board = state.board
        n = board.size()
        empty_positions = [(i,j) for i in range(n) for j in range(n) if board.get_number(i,j) == 2]

        # no valid moves or invalid board
        if not empty_positions or not board.is_valid(): return []

        # check for obvious moves (if current position is empty, and the 2 adjacent are equal, the current is different)
        for row in range(n):
            for col in range(n):
                if board.get_number(row,col) == 2:
                    
                    #vertical
                    up, down = board.adjacent_vertical_numbers(row, col)
                    if up == down == 0 :
                        return [(row,col,1)]
                    elif up == down == 1:
                        return [(row, col, 0)]

                    #horizontal
                    left, right = board.adjacent_horizontal_numbers(row, col)
                    if left == right == 0 :
                        return [(row,col,1)]
                    elif left == right == 1:
                        return [(row, col, 0)]
        
        # check for obvious moves (if 2 adjacent are equal, the empty position adjacent to them is different)
        for row in range(n):
            for col in range(n):
                if board.get_number(row,col) != 2: 
                    cur = board.get_number(row,col)

                    #vertical
                    up, down = board.adjacent_vertical_numbers(row, col)
                    if up == 2 and down == cur == 0:
                        if row-1>=0: return [(row-1,col,1)]
                    if up == 2 and down == cur == 1:
                        if row-1>=0: return [(row-1,col,0)]

                    if down == 2 and up == cur == 0:
                        if row+1<n: return [(row+1,col,1)]
                    if down == 2 and up == cur == 1:
                        if row+1<n: return [(row+1,col,0)]

                    #horizontal
                    left, right = board.adjacent_horizontal_numbers(row, col)
                    if left == 2 and right == cur == 0:
                        if col-1>=0: return [(row,col-1,1)]
                    if left == 2 and right == cur == 1:
                        if col-1>=0: return [(row,col-1,0)]

                    if right == 2 and left == cur == 0:
                        if col+1<n: return [(row,col+1,1)]
                    if right == 2 and left == cur == 1:
                        if col+1<n: return [(row,col+1,0)]

        #check for obvious moves (if there's a row with enough 0s or 1s (half), then all empty positions in that row should be different)
        for row in range(n):
            ones=board.grid[row].count(1)
            zeros=board.grid[row].count(0)
            if n%2==0:
                if ones==n//2:
                    for col in range(n):
                        if board.get_number(row,col)==2: return [(row,col,0)]
                elif zeros==n//2:
                    for col in range(n):
                        if board.get_number(row,col)==2: return [(row,col,1)]
            else:
                if ones==n//2+1:
                    for col in range(n):
                        if board.get_number(row,col)==2: return [(row,col,0)]
                elif zeros==n//2+1:
                    for col in range(n):
                        if board.get_number(row,col)==2: return [(row,col,1)]


        #check for obvious moves (if there's a col with enough 0s or 1s (half), then all empty positions in that col should be different)
        cols = list(zip(*board.grid))

        for col in range(n):
            ones=cols[col].count(1)
            zeros=cols[col].count(0)
            if n%2==0:
                if ones==n//2:
                    for row in range(n):
                        if board.get_number(row,col)==2: return [(row,col,0)]
                elif zeros==n//2:
                    for row in range(n):
                        if board.get_number(row,col)==2: return [(row,col,1)]
            else:
                if ones==n//2+1:
                    for row in range(n):
                        if board.get_number(row,col)==2: return [(row,col,0)]
                elif zeros==n//2+1:
                    for row in range(n):
                        if board.get_number(row,col)==2: return [(row,col,1)]

        #other available positions
        for x,y in empty_positions:
            actions.append((x,y,0))
            actions.append((x,y,1))

        return actions


    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        board = state.board
        n = board.size()

        new_board = Board(n)
        for i in range(n):
            for j in range(n):
                value = board.get_number(i,j)
                new_board.set_number(i,j,value)
        
        x,y,new_val = action
        new_board.set_number(x,y,new_val)
        new_state = TakuzuState(new_board)
        return new_state


    def goal_test(self, state: TakuzuState):
        board = state.board
        n = board.size()
        for i in range(n):
            for j in range(n):
                if board.get_number(i,j) == 2: return False
        return board.is_valid()

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        #heuristica escolhida: contar numero rows cheias e cols cheias, retornar 2*n-(rows_cheias+cols_cheias)
        rows_completed = cols_completed = 0
        grid = node.state.board.grid
        n = node.state.board.size()
        for row in grid:
            rows_completed += row.count(2) == 0
        for col in list(zip(*grid)):
            cols_completed += col.count(2) == 0
        return 2*n - rows_completed - cols_completed



if __name__ == "__main__":
    # Ler o ficheiro de input de sys.argv[1],
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.

    board = Board.parse_instance_from_stdin()
    problem = Takuzu(board)
    goal_node = depth_first_tree_search(problem)
    print(goal_node.state.board.print_board(), end = '')
