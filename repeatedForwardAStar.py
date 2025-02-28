from aStarBasic import a_star_search

def manhattan_distance(a, b):
    """
    Computes the Manhattan distance between points a and b.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def repeated_forward_a_star(grid, starts, goals):
    """
    Repeated Forward A* algorithm.
    The agent uses a known grid that is updated as it observes adjacent cells.
    The agent uses a known start and goal positions.
    The agent uses a known grid that is updated as it observes adjacent cells.
    Returns the path (as a list of coordinates) and total number of expanded cells.
    """