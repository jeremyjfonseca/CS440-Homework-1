import numpy as np
import random
import matplotlib.pyplot as plt

def generate_grid(n):
    """
    Generate a single n x n gridworld using DFS.
    0 = unblocked, 1 = blocked.
    """
    def get_neighbors(x, y):
        """Get the list of valid neighbors for a cell (x, y)."""
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < n - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < n - 1:
            neighbors.append((x, y + 1))
        random.shuffle(neighbors)
        return neighbors

    def dfs(x, y):
        """Depth-first search to generate the grid."""
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if grid[cx][cy] == 0:  # If the cell is already visited, skip it
                continue
            grid[cx][cy] = 0  # Mark the cell as visited (unblocked)
            neighbors = get_neighbors(cx, cy)
            for nx, ny in neighbors:
                if grid[nx][ny] == 1:  # If the neighbor is unvisited (blocked)
                    stack.append((nx, ny))

    # Initialize the grid with all cells blocked
    grid = [[1 if random.random() < 0.3 else 0 for _ in range(n)] for _ in range(n)]
    
    # Start DFS from the center of the grid
    start_x, start_y = n // 2, n // 2
    dfs(start_x, start_y)
    
    return grid

def visualize_grid(grid):
    """Display a grid using matplotlib with gridlines."""
    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(grid), cmap='binary')
    plt.title("Generated Gridworld")
    plt.axis('off')

    plt.show()

def generate_multiple_grids(num_grids, size):
    """
    Generate multiple gridworlds.
    
    Parameters:
    num_grids (int): Number of grids to generate.
    size (int): Size of each grid (must be an odd number).
    
    Returns:
    list: A list of generated grids.
    """
    grids = []
    for _ in range(num_grids):
        grids.append(generate_grid(size))
    return grids

if __name__ == "__main__":
    n = 21  # Size of the gridworld (must be an odd number)
    gridworld = generate_multiple_grids( 10,  n)
    for i in range(5):
        visualize_grid(gridworld[i])
