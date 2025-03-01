import heapq
import numpy as np

class AdaptiveAStar:
    def __init__(self, grid, start, goal):
        self.grid = np.array(grid)
        self.start = start
        self.goal = goal
        self.n = grid.shape[0]  # Grid size

        # Initialize search-related variables
        self.g = np.full((self.n, self.n), float('inf'))  # Cost-to-come values
        self.h = np.zeros((self.n, self.n))  # h-values (heuristics), initially zero
        self.search = np.zeros((self.n, self.n), dtype=int)  # Tracks search iterations
        self.tree = np.full((self.n, self.n, 2), -1)  # Stores back-pointers for path reconstruction
        self.counter = 0  # Search iteration counter
        self.expanded_cells = 0  # Counter for expanded cells

        # Movement directions (North, South, West, East)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Initialize the heuristic using Manhattan distance
        for x in range(self.n):
            for y in range(self.n):
                self.h[x, y] = self.heuristic((x, y))

    def heuristic(self, cell):
        """Compute Manhattan distance heuristic to the goal."""
        return abs(cell[0] - self.goal[0]) + abs(cell[1] - self.goal[1])

    def compute_path(self, open_list, closed_list):
        """Implements the ComputePath() function and tracks expanded cells."""
        while open_list:
            _, neg_g_value, current = heapq.heappop(open_list)  # Get state with smallest f-value
            cx, cy = current

            if (cx, cy) == self.goal:
                return  # Goal reached

            if (cx, cy) in closed_list:
                continue  # Skip already expanded nodes

            closed_list.add((cx, cy))
            self.expanded_cells += 1  # Track expanded cells

            # Expand neighbors
            for dx, dy in self.directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.n and 0 <= ny < self.n and self.grid[nx, ny] == 0:  # Valid and unblocked
                    if self.search[nx, ny] < self.counter:
                        self.g[nx, ny] = float('inf')
                        self.search[nx, ny] = self.counter

                    new_cost = self.g[cx, cy] + 1  # Cost of moving to neighbor
                    if new_cost < self.g[nx, ny]:
                        self.g[nx, ny] = new_cost
                        self.tree[nx, ny] = [cx, cy]  # Back-pointer
                        heapq.heappush(open_list, (self.g[nx, ny] + self.h[nx, ny], -self.g[nx, ny], (nx, ny)))

    def update_h_values(self):
        """Update h-values for states expanded in the last search."""
        goal_g_value = self.g[self.goal]
        for x in range(self.n):
            for y in range(self.n):
                if self.search[x, y] == self.counter:  # If state was expanded in last search
                    self.h[x, y] = goal_g_value - self.g[x, y]  # Update h-value

    def find_path(self):
        """Main function to execute Adaptive A* and return expanded cells count."""
        self.expanded_cells = 0  # Reset expanded cells count before each search
        
        while self.start != self.goal:
            self.counter += 1
            sx, sy = self.start
            self.g[sx, sy] = 0
            self.search[sx, sy] = self.counter
            self.g[self.goal] = float('inf')
            self.search[self.goal] = self.counter
            
            open_list = []
            closed_list = set()
            
            # Insert start state into OPEN list
            heapq.heappush(open_list, (self.g[self.start] + self.h[self.start], 0, self.start))

            # Compute shortest presumed-unblocked path
            self.compute_path(open_list, closed_list)

            if not open_list:
                print("I cannot reach the target.")
                return None, self.expanded_cells

            # Follow path from goal to start
            path = []
            cx, cy = self.goal
            while (cx, cy) != self.start:
                path.append((cx, cy))
                cx, cy = tuple(self.tree[cx, cy])

            path.append(self.start)
            path.reverse()

            # Update heuristic values for next search
            self.update_h_values()

            # Move agent along the path until an obstacle is encountered
            for cell in path:
                self.start = cell
                if cell == self.goal:
                    print("I reached the target.")
                    return path, self.expanded_cells
                
                # If a neighboring cell is found to be blocked, update the grid
                for dx, dy in self.directions:
                    nx, ny = self.start[0] + dx, self.start[1] + dy
                    if 0 <= nx < self.n and 0 <= ny < self.n and self.grid[nx, ny] == 1:
                        self.g[nx, ny] = float('inf')  # Mark it as unreachable

        return None, self.expanded_cells  # In case something goes wrong

if __name__ == "__main__":
    # Example Usage
    from gridworldGeneration import generate_grid, visualize_grid

    size = 101
    grid = np.array(generate_grid(size))
    start = (0, 0)
    goal = (size - 1, size - 1)

    print("Initial Grid:")
    visualize_grid(grid)

    solver = AdaptiveAStar(grid, start, goal)
    path, expanded_cells = solver.find_path()

    if path:
        print("Path found!")
        print(f"Number of expanded cells: {expanded_cells}")
    else:
        print("No path to target.")
        print(f"Number of expanded cells: {expanded_cells}")
