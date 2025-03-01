import heapq
import numpy as np

class RepeatedForwardAStar:
    def __init__(self, grid, start, goal):
        self.grid = np.array(grid)
        self.start = start
        self.goal = goal
        self.n = grid.shape[0]  # Grid size
        
        self.g = np.full((self.n, self.n), float('inf'))  
        self.search = np.zeros((self.n, self.n), dtype=int)  
        self.tree = np.full((self.n, self.n, 2), -1)  
        self.counter = 0  
        self.expanded_cells = 0  

        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(self, cell):
        return abs(cell[0] - self.goal[0]) + abs(cell[1] - self.goal[1])

    def compute_path(self, open_list, closed_list):
        while open_list:
            _, neg_g_value, current = heapq.heappop(open_list)
            cx, cy = current

            if (cx, cy) == self.goal:
                return  

            if (cx, cy) in closed_list:
                continue  

            closed_list.add((cx, cy))
            self.expanded_cells += 1  

            for dx, dy in self.directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.n and 0 <= ny < self.n and self.grid[nx, ny] == 0:  
                    if self.search[nx, ny] < self.counter:
                        self.g[nx, ny] = float('inf')
                        self.search[nx, ny] = self.counter

                    new_cost = self.g[cx, cy] + 1  
                    if new_cost < self.g[nx, ny]:
                        self.g[nx, ny] = new_cost
                        self.tree[nx, ny] = [cx, cy]  
                        heapq.heappush(open_list, (self.g[nx, ny] + self.heuristic((nx, ny)), -self.g[nx, ny], (nx, ny)))

    def find_path(self):
        self.expanded_cells = 0  
        
        while self.start != self.goal:
            self.counter += 1
            sx, sy = self.start
            self.g[sx, sy] = 0
            self.search[sx, sy] = self.counter
            self.g[self.goal] = float('inf')
            self.search[self.goal] = self.counter
            
            open_list = []
            closed_list = set()
            
            heapq.heappush(open_list, (self.heuristic(self.start), 0, self.start))

            self.compute_path(open_list, closed_list)

            if not open_list:
                return None, self.expanded_cells

            path = []
            cx, cy = self.goal
            while (cx, cy) != self.start:
                path.append((cx, cy))
                cx, cy = tuple(self.tree[cx, cy])

            path.append(self.start)
            path.reverse()

            for cell in path:
                self.start = cell
                if cell == self.goal:
                    return path, self.expanded_cells
                
        return None, self.expanded_cells
