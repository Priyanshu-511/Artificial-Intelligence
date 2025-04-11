import heapq

def print_solution(board):
    """Prints the chessboard solution and positions of all queens."""
    positions = []
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col]:
                positions.append((row, col))
    
    for row in board:
        print(" ".join("Q" if col else "." for col in row))
    print("Queen Positions:", positions)
    print("\n")

def heuristic(board, n):
    """Alternative heuristic function: Counts number of safe positions left for future queens."""
    safe_positions = 0
    for row in range(n):
        for col in range(n):
            if not board[row][col]:  # Empty position
                if is_safe(board, row, col, n):
                    safe_positions += 1
    return n**2 - safe_positions  # Lower is better (fewer conflicts)

def is_safe(board, row, col, n):
    """Checks if placing a queen at board[row][col] is safe by verifying column and diagonal constraints."""
    # Check column
    for i in range(row):
        if board[i][col]:
            return False
    
    # Check upper-left diagonal
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if board[i][j]:
            return False
        i -= 1
        j -= 1
    
    # Check upper-right diagonal
    i, j = row - 1, col + 1
    while i >= 0 and j < n:
        if board[i][j]:
            return False
        i -= 1
        j += 1
    
    return True

def a_star_n_queens(n):
    """A* search algorithm to solve N-Queens problem."""
    open_list = []
    heapq.heappush(open_list, (0, [], 0))  # (f-cost, board_state, row)
    nodeExpand = 0
    
    while open_list:
        _, board_state, row = heapq.heappop(open_list)
        nodeExpand+=1
        
        if row == n:
            board = [[0] * n for _ in range(n)]
            for r, c in board_state:
                board[r][c] = 1
            print_solution(board)
            print(nodeExpand)
            return
        
        for col in range(n):
            if is_safe([[1 if (r, c) in board_state else 0 for c in range(n)] for r in range(n)], row, col, n):
                new_state = board_state + [(row, col)]
                g_cost = row + 1  # Number of placed queens
                h_cost = heuristic([[1 if (r, c) in new_state else 0 for c in range(n)] for r in range(n)], n)
                f_cost = g_cost + h_cost
                heapq.heappush(open_list, (f_cost, new_state, row + 1))
    print(nodeExpand)
    print("No solution found")

# Run the 8-Queens solver using A*
a_star_n_queens(8)
