import numpy as np
import time
import imageio
import matplotlib.pyplot as plt
import os
from gridworldGeneration import generate_grid
from repeated_forward_astar import RepeatedForwardAStar
from repeated_backward_astar import RepeatedBackwardAStar
from adaptive_astar import AdaptiveAStar

def visualize_grid(grid, agent, visited, presumed_path, start, goal, filename):
    """Generates and saves a single frame of the grid for visualization."""
    n = grid.shape[0]
    img = np.full((n, n, 3), 255, dtype=np.uint8)  # Create a white grid

    # Color settings
    obstacle_color = [0, 0, 0]  # Black (blocked cells)
    visited_color = [200, 255, 200]  # Light green (visited path)
    presumed_path_color = [100, 255, 100]  # Bright green (presumed shortest path)
    agent_color = [255, 0, 0]  # Red (agent location)
    start_color = [0, 0, 255]  # Blue (start state)
    goal_color = [0, 255, 0]  # Green (goal state)

    # Draw the grid
    for x in range(n):
        for y in range(n):
            if grid[x, y] == 1:  # Blocked cell
                img[x, y] = obstacle_color
            elif (x, y) in visited:  # Show visited cells
                img[x, y] = visited_color
            elif (x, y) in presumed_path:  # Show presumed shortest path at this step
                img[x, y] = presumed_path_color

    # Draw agent, start, and goal
    img[start] = start_color
    img[goal] = goal_color
    img[agent] = agent_color  # Agent's current position

    # Save the frame
    plt.imsave(filename, img)

def run_experiment(grid, grid_size, algorithm_class, algorithm_name):
    """Runs a search algorithm and generates a GIF visualizing the search process."""
    start, goal = (0, 0), (grid_size - 1, grid_size - 1)
    
    # Copy the grid so each search algorithm uses the same environment
    grid_copy = np.copy(grid)

    # Initialize the search algorithm
    solver = algorithm_class(grid_copy, start, goal)

    frames = []  # Store frame filenames
    visited = set()  # Track visited cells

    # Run the search while capturing frames
    while solver.start != solver.goal:
        presumed_path, _ = solver.find_path()  # Get the presumed shortest path

        if presumed_path is None:
            print(f"{algorithm_name} could not find a path!")
            return

        for agent_pos in presumed_path:
            visited.add(agent_pos)  # Track where the agent has been

            # Capture the agent's presumed path at this step
            frame_filename = f"{algorithm_name}_frame_{len(frames)}.png"
            visualize_grid(grid_copy, agent_pos, visited, presumed_path, start, goal, frame_filename)
            frames.append(frame_filename)

    # Generate GIF
    gif_filename = f"{algorithm_name}_search_animation.gif"
    with imageio.get_writer(gif_filename, mode='I', duration=0.2) as writer:
        for filename in sorted(frames, key=lambda x: int(x.split('_')[-1].split('.')[0])):  # Sort frames in order
            image = imageio.imread(filename)
            writer.append_data(image)

    # Cleanup images
    for filename in frames:
        os.remove(filename)

    print(f" GIF saved: {gif_filename}")

if __name__ == "__main__":
    grid_size = 101  # Size of the grid

    # Generate a valid grid where start and goal are unblocked
    while True:
        grid = np.array(generate_grid(grid_size))
        start, goal = (0, 0), (grid_size - 1, grid_size - 1)
        if grid[start] == 0 and grid[goal] == 0:  # Ensure start and goal are unblocked
            break

    print(" Grid generated. Running all three search algorithms...")

    # Run each algorithm using the **same grid**
    run_experiment(grid, grid_size, RepeatedForwardAStar, "Repeated_Forward_AStar")
    run_experiment(grid, grid_size, RepeatedBackwardAStar, "Repeated_Backward_AStar")
    run_experiment(grid, grid_size, AdaptiveAStar, "Adaptive_AStar")
