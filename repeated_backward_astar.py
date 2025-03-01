import heapq
import numpy as np

class RepeatedBackwardAStar:
    def __init__(self, grid, start, goal):
        self.grid = np.array(grid)
        self.start = start
        self.goal = goal
        self.n = grid.shape[0]  # Grid size
        
        self.g = np.full((self.n, self.n), float('inf'))  # Cost-to-come values
        self.search = np.zeros((self.n, self.n), dtype=int)  # Tracks when a state was last searched
        self.tree = np.full((self.n, self.n, 2), -1)  # Stores back-pointers for path reconstruction
        self.counter = 0  # Search iteration counter
        self.expanded_cells = 0  # Counter for expanded cells

        # Movement directions (North, South, West, East)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(self, cell):
        """Compute Manhattan distance heuristic from the start (since search is backward)."""
        return abs(cell[0] - self.start[0]) + abs(cell[1] - self.start[1])

    def compute_path(self, open_list, closed_list):
        """Implements the ComputePath() function for backward search and tracks expanded cells."""
        while open_list:
            _, neg_g_value, current = heapq.heappop(open_list)  # Get state with smallest f-value
            cx, cy = current

            if (cx, cy) == self.start:
                return  # Search has reached the start

            if (cx, cy) in closed_list:
                continue  # Skip already expanded nodes

            closed_list.add((cx, cy))
            self.expanded_cells += 1  # Track expanded cells

            # Expand neighbors (backward)
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
                        heapq.heappush(open_list, (self.g[nx, ny] + self.heuristic((nx, ny)), -self.g[nx, ny], (nx, ny)))

    def find_path(self):
        """Main function to execute Repeated Backward A* and return expanded cells count."""
        self.expanded_cells = 0  # Reset expanded cells count before each search
        
        while self.goal != self.start:
            self.counter += 1
            gx, gy = self.goal
            self.g[gx, gy] = 0
            self.search[gx, gy] = self.counter
            self.g[self.start] = float('inf')
            self.search[self.start] = self.counter
            
            open_list = []
            closed_list = set()
            
            # Insert goal state into OPEN list
            heapq.heappush(open_list, (self.heuristic(self.goal), 0, self.goal))

            # Compute shortest presumed-unblocked path (backward search)
            self.compute_path(open_list, closed_list)

            if not open_list:
                print("I cannot reach the start.")
                return None, self.expanded_cells

            # Follow path from start to goal (since we searched backward)
            path = []
            cx, cy = self.start
            while (cx, cy) != self.goal:
                path.append((cx, cy))
                cx, cy = tuple(self.tree[cx, cy])

            path.append(self.goal)
            path.reverse()

            # Move agent along the path until an obstacle is encountered
            for cell in path:
                self.goal = cell
                if cell == self.start:
                    print("I reached the start.")
                    return path, self.expanded_cells
                
                # If a neighboring cell is found to be blocked, update the grid
                for dx, dy in self.directions:
                    nx, ny = self.goal[0] + dx, self.goal[1] + dy
                    if 0 <= nx < self.n and 0 <= ny < self.n and self.grid[nx, ny] == 1:
                        self.g[nx, ny] = float('inf')  # Mark it as unreachable

        return None, self.expanded_cells  # In case something goes wrong
