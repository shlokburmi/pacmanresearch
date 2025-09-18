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
add an agent to the game, connect it to the PacmanGame class through the
`pacman.py` file.  Note that this file is imported by `pacman.py` and is not a
stand-alone script.
"""

from game import Directions, Agent, Actions
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

#######################################################
# This portion is written for you, but will need to be #
# changed trivially to solve the search problem.      #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs depth-first search on a
    PositionSearchProblem to find location (1,1)

    Options for fn include:
      depth_first_search or dfs
      breadth_first_search or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depth_first_search', prob='PositionSearchProblem', heuristic='null_heuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print(('[SearchAgent] using function ' + fn))
            self.search_function = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in search_agents.py or search.py.')
            print(('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic)))
            # Store the search function (substituting lambda for create_function)
            self.search_function = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.search_type = globals()[prob]
        print(('[SearchAgent] using problem type ' + prob))

    def register_initial_state(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.search_function == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.search_type(state) # Makes a new search problem
        self.actions  = self.search_function(problem) # Find a path
        total_cost = problem.get_cost_of_actions(self.actions)
        print(('Path found with total cost of %d in %.1f seconds' % (total_cost, time.time() - starttime)))
        if '_expanded' in dir(problem): print(('Search nodes expanded: %d' % problem._expanded))

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
        "Stores the start and goal. The start is stored in the observation."
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
        if actions == None: return 999999
        x,y= self.get_start_state()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its legal
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += self.cost_fn((x,y))
        return cost

class FoodSearchProblem(search.SearchProblem):
    """
    A search problem where the goal is to eat all of the food.
    A state is represented as a tuple: (pacmanPosition, foodGrid)
    """
    def __init__(self, starting_game_state):
        self.start = (starting_game_state.get_pacman_position(), starting_game_state.get_food())
        self.walls = starting_game_state.get_walls()
        self._expanded = 0
        self.heuristic_info = {}

    def get_start_state(self):
        return self.start

    def is_goal_state(self, state):
        return state[1].count() == 0

    def get_successors(self, state):
        successors = []
        self._expanded += 1
        pos, food_grid = state
        x, y = pos

        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.direction_to_vector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                next_food = food_grid.copy()
                next_food[nextx][nexty] = False
                successors.append((((nextx, nexty), next_food), direction, 1))
        return successors

    def get_cost_of_actions(self, actions):
        x, y = self.get_start_state()[0]
        cost = 0
        for action in actions:
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class FoodSearchAgent(SearchAgent):
    """
    An agent that uses A* search to find a path to eat all the food.
    """
    def __init__(self):
        self.search_function = lambda prob: search.a_star_search(prob, food_heuristic)
        self.search_type = FoodSearchProblem
        
def food_heuristic(state, problem):
    """
    Heuristic for the FoodSearchProblem. It calculates the Manhattan distance
    from the current position to the farthest food dot.
    """
    position, food_grid = state
    food_list = food_grid.as_list()
    if not food_list:
        return 0
    
    # A simple but effective heuristic is the distance to the farthest food dot.
    # This is admissible because Pac-Man must travel at least that far.
    max_dist = 0
    for food_pos in food_list:
        dist = util.manhattan_distance(position, food_pos)
        if dist > max_dist:
            max_dist = dist
    return max_dist

def manhattan_heuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def register_initial_state(self, state):
        self.actions = []
        self.action_index = 0 # Reset action index
        problem = self.search_type(state)
        self.actions = self.search_function(problem)

    def get_action(self, state):
        if self.action_index >= len(self.actions):
             # If out of actions, find the next closest dot and re-plan
            food_grid = state.get_food()
            if food_grid.count() == 0:
                return Directions.STOP

            pacman_pos = state.get_pacman_position()
            food_list = food_grid.as_list()
            closest_dot = min(food_list, key=lambda food: util.manhattan_distance(pacman_pos, food))
            
            problem = PositionSearchProblem(state, start=pacman_pos, goal=closest_dot, warn=False, visualize=False)
            
            # Here you can use any search function. Let's use A* as a default.
            self.actions = search.a_star_search(problem, heuristic=manhattan_heuristic)
            self.action_index = 0
            
            if not self.actions:
                 return Directions.STOP

        action = self.actions[self.action_index]
        self.action_index += 1
        return action
