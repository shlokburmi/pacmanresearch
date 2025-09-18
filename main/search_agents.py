# search_agents.py
# ----------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depth_first_search

Commands deeply nested in folders might require slightly different commands to
run.
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def get_action(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.get_legal_pacman_actions():
            return Directions.WEST
        else:
            return Directions.STOP

def null_heuristic(state, problem=None):
    """
    A heuristic function that returns 0 for any state.
    """
    return 0

def manhattan_heuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


#######################################################
# This portion is written for you, but will be useful #
# for your project.                                    #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs depth first search on a PositionSearchProblem
    to find location (1,1)

    Options for fn include:
      depth_first_search or dfs
      breadth_first_search or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depth_first_search', prob='PositionSearchProblem', heuristic='null_heuristic', **kwargs):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            self.search_function = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in search_agents.py or search.py.')
            self.search_function = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in search_agents.py.')
        self.search_type = globals()[prob]
        
        # Store extra problem arguments
        self.problem_args = kwargs

    def register_initial_state(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.search_function == None: raise Exception("No search function provided for SearchAgent")
        
        # Pass the extra arguments to the search problem constructor
        problem = self.search_type(state, **self.problem_args) 
        self.actions  = self.search_function(problem) # Find a path

    def get_action(self, state):
        """
        Returns the next action in the path chosen earlier (in
        register_initial_state).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'action_index' not in dir(self): self.action_index = 0
        i = self.action_index
        self.action_index += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, game_state, cost_fn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        game_state: A GameState object (pacman.py)
        cost_fn: A function from a search state (position) to a non-negative number
        goal: A position in the game_state
        """
        self.walls = game_state.get_walls()
        self.start_state = game_state.get_pacman_position()
        if start != None: self.start_state = start
        self.goal = goal
        self.cost_fn = cost_fn
        self.visualize = visualize
        if warn and (game_state.get_num_food() > 1 or game_state.has_scared_ghost()):
            print('Warning: this does not find a short path to eat all the food.')
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def get_start_state(self):
        return self.start_state

    def is_goal_state(self, state):
        is_goal = state == self.goal
        return is_goal

    def get_successors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                next_state = (nextx, nexty)
                cost = self.cost_fn(next_state)
                successors.append( ( next_state, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def get_cost_of_actions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.get_start_state()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its legal
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.cost_fn((x,y))
        return cost

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacman_position, food_grid ) where
      pacman_position: a tuple (x,y) of integers specifying Pacman's position
      food_grid:       a Grid (see game.py) of booleans, where food_grid[x][y] is True if there is food at (x,y)
    """
    def __init__(self, starting_game_state):
        self.start = (starting_game_state.get_pacman_position(), starting_game_state.get_food())
        self.walls = starting_game_state.get_walls()
        self.starting_game_state = starting_game_state
        self._expanded = 0 # DO NOT CHANGE
        self.heuristic_info = {} # A dictionary for the heuristic to store information

    def get_start_state(self):
        return self.start

    def is_goal_state(self, state):
        return state[1].count() == 0

    def get_successors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.direction_to_vector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                next_food = state[1].copy()
                next_food[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), next_food), direction, 1) )
        return successors

    def get_cost_of_actions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.get_start_state()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

def food_heuristic(state, problem):
    position, food_grid = state
    food_list = food_grid.as_list()
    if not food_list:
        return 0
    # Find the Manhattan distance to the farthest food dot
    max_dist = 0
    for food in food_list:
        dist = util.manhattan_distance(position, food)
        if dist > max_dist:
            max_dist = dist
    return max_dist

class FoodSearchAgent(SearchAgent):
    """
    An agent that uses A* search to find a path to eat all the food.
    """
    def __init__(self):
        self.search_function = lambda prob: search.a_star_search(prob, food_heuristic)
        self.search_type = FoodSearchProblem
        
class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def register_initial_state(self, state):
        self.actions = []
        current_state = state
        while(current_state.get_food().count() > 0):
            start_point = current_state.get_pacman_position()
            food_grid = current_state.get_food()
            food_list = food_grid.as_list()
            closest_dot = min(food_list, key=lambda x: util.manhattan_distance(start_point, x))
            prob = PositionSearchProblem(current_state, start=start_point, goal=closest_dot, warn=False, visualize=False)
            path = self.search_function(prob)
            self.actions += path
            # Update the state to reflect the moves
            for action in path:
                legal = current_state.get_legal_actions()
                if action not in legal:
                    # This can happen if a ghost is in the way
                    # In a real game, we'd need more complex logic
                    # For this problem, we'll just stop planning
                    return
                current_state = current_state.generate_successor(0, action)
        self.action_index = 0

# This is the agent you were trying to create.
# It uses bidirectional A* search on a PositionSearchProblem.
class BidirectionalAgent(SearchAgent):
    def __init__(self, **kwargs):
        # We pass fn, prob, and heuristic directly to the SearchAgent constructor
        super().__init__(fn='bidirectional_astar', prob='PositionSearchProblem', heuristic='manhattan_heuristic', **kwargs)

