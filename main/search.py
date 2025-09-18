# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding tools were added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terms: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, step_cost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'step_cost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tiny_maze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tiny_maze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.
    """
    from util import Stack
    fringe = Stack()
    fringe.push((problem.get_start_state(), []))
    visited = set()

    while not fringe.is_empty():
        node, actions = fringe.pop()
        if problem.is_goal_state(node):
            return actions
        if node not in visited:
            visited.add(node)
            for successor, action, cost in problem.get_successors(node):
                new_actions = actions + [action]
                fringe.push((successor, new_actions))
    return []

def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue
    fringe = Queue()
    fringe.push((problem.get_start_state(), []))
    visited = set()
    visited.add(problem.get_start_state())

    while not fringe.is_empty():
        node, actions = fringe.pop()
        if problem.is_goal_state(node):
            return actions
        for successor, action, cost in problem.get_successors(node):
            if successor not in visited:
                visited.add(successor)
                new_actions = actions + [action]
                fringe.push((successor, new_actions))
    return []

def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue
    fringe = PriorityQueue()
    fringe.push((problem.get_start_state(), []), 0)
    visited = {}

    while not fringe.is_empty():
        node, actions = fringe.pop()
        if problem.is_goal_state(node):
            return actions
        
        cost = problem.get_cost_of_actions(actions)
        if node not in visited or cost < visited[node]:
            visited[node] = cost
            for successor, action, step_cost in problem.get_successors(node):
                new_actions = actions + [action]
                new_cost = problem.get_cost_of_actions(new_actions)
                fringe.push((successor, new_actions), new_cost)
    return []

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from util import PriorityQueue
    fringe = PriorityQueue()
    start_state = problem.get_start_state()
    fringe.push((start_state, []), heuristic(start_state, problem))
    visited = {}

    while not fringe.is_empty():
        node, actions = fringe.pop()
        
        cost_so_far = problem.get_cost_of_actions(actions)
        
        state_to_check = node if not isinstance(node, tuple) else (node[0], tuple(node[1].as_list()))

        if state_to_check in visited and visited[state_to_check] <= cost_so_far:
            continue
            
        visited[state_to_check] = cost_so_far

        if problem.is_goal_state(node):
            return actions

        for successor, action, step_cost in problem.get_successors(node):
            new_actions = actions + [action]
            g_cost = problem.get_cost_of_actions(new_actions)
            h_cost = heuristic(successor, problem)
            f_cost = g_cost + h_cost
            fringe.push((successor, new_actions), f_cost)
            
    return []

def bidirectional_astar_search(problem, heuristic=null_heuristic):
    """
    Bidirectional A* search. This implementation is best suited for problems
    with a single, explicitly defined goal state.
    """
    from util import PriorityQueue

    start_node = problem.get_start_state()

    # Ensure the problem has a single, defined goal for this search to work
    if not hasattr(problem, 'goal'):
        print("Warning: Bidirectional search requires a single 'goal' attribute. Falling back to A*.")
        return a_star_search(problem, heuristic)

    goal_node = problem.goal
        
    if problem.is_goal_state(start_node):
        return []

    # --- Forward Search Initialization ---
    f_fringe = PriorityQueue()
    f_fringe.push(start_node, heuristic(start_node, problem))
    f_g_costs = {start_node: 0}
    f_paths = {start_node: []}
    f_closed = set()

    # --- Backward Search Initialization ---
    b_fringe = PriorityQueue()
    # Backward heuristic estimates cost from a node to the start_node
    b_heuristic = lambda state, prob: util.manhattan_distance(state, start_node)
    b_fringe.push(goal_node, b_heuristic(goal_node, problem))
    b_g_costs = {goal_node: 0}
    b_paths = {goal_node: []}
    b_closed = set()

    # Best path found so far
    mu = float('inf') 
    best_path = []

    def _reverse_path(path):
        """Reverses a path and its actions."""
        rev_path = []
        reverse_map = {"North": "South", "South": "North", "East": "West", "West": "East", "Stop": "Stop"}
        for action in reversed(path):
            rev_path.append(reverse_map.get(action))
        return rev_path

    while not f_fringe.is_empty() and not b_fringe.is_empty():

        # Check for intersection and update best path
        for node in f_g_costs:
            if node in b_g_costs:
                new_cost = f_g_costs[node] + b_g_costs[node]
                if new_cost < mu:
                    mu = new_cost
                    best_path = f_paths[node] + _reverse_path(b_paths[node])

        # Termination condition
        if f_fringe.peek()[1] + b_fringe.peek()[1] >= mu:
            return best_path

        # --- Expand Forward ---
        if f_fringe.peek()[1] < b_fringe.peek()[1]:
            u, _ = f_fringe.pop()
            f_closed.add(u)
            
            for v, action, cost in problem.get_successors(u):
                new_g = f_g_costs[u] + cost
                if v not in f_g_costs or new_g < f_g_costs[v]:
                    f_g_costs[v] = new_g
                    f_paths[v] = f_paths[u] + [action]
                    f_fringe.update(v, new_g + heuristic(v, problem))
        
        # --- Expand Backward ---
        else:
            u, _ = b_fringe.pop()
            b_closed.add(u)

            # Note: We need successors to the node `u` for backward search.
            # This requires the problem to handle successor generation symmetrically.
            for v, action, cost in problem.get_successors(u):
                new_g = b_g_costs[u] + cost
                if v not in b_g_costs or new_g < b_g_costs[v]:
                    b_g_costs[v] = new_g
                    b_paths[v] = b_paths[u] + [action]
                    b_fringe.update(v, new_g + b_heuristic(v, problem))

    return best_path


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
bidirectional_astar = bidirectional_astar_search

