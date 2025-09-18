from __future__ import print_function
from builtins import range
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses actions based on an evaluation function.
    """
    def get_action(self, game_state):
        legal_moves = game_state.get_legal_actions()
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices)
        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        successor_game_state = current_game_state.generate_pacman_successor(action)
        return successor_game_state.get_score()

def score_evaluation_function(current_game_state):
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        self.index = 0  # Pacman is always agent 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth)

def bidirectionalAStarSearch(problem, heuristic=None):
    if heuristic is None:
        heuristic = lambda s, p: 0

    from util import PriorityQueue, Queue
    start = problem.get_start_state()
    if problem.is_goal_state(start):
        return []
    # Efficiently find goal
    goal = getattr(problem, 'goal', None)
    if goal is None:
        q = Queue()
        q.push(start)
        visited = set([start])
        while not q.is_empty():
            state = q.pop()
            if problem.is_goal_state(state):
                goal = state
                break
            for succ, _, _ in problem.get_successors(state)[:4]:  # limit successors
                if succ not in visited:
                    visited.add(succ)
                    q.push(succ)
        if goal is None:
            return []

    frontier_fwd = PriorityQueue()
    frontier_bwd = PriorityQueue()
    frontier_fwd.push((start, []), heuristic(start, problem))
    frontier_bwd.push((goal, []), heuristic(goal, problem))
    visited_fwd = {start: 0}
    visited_bwd = {goal: 0}
    path_fwd = {start: []}
    path_bwd = {goal: []}
    best_cost = float('inf')
    best_path = []
    mu = float('inf')

    while not frontier_fwd.is_empty() and not frontier_bwd.is_empty():
        # Check termination
        f_min = frontier_fwd.heap[0][0] if frontier_fwd.heap else float('inf')
        b_min = frontier_bwd.heap[0][0] if frontier_bwd.heap else float('inf')
        if min(f_min, b_min) >= mu:
            break

        # Expand forward frontier
        curr, path = frontier_fwd.pop()
        g = problem.get_cost_of_actions(path)
        if curr in visited_bwd:
            total_cost = g + visited_bwd[curr]
            if total_cost < best_cost:
                best_cost = total_cost
                rev_path = path_bwd[curr][::-1]
                # Reverse backward path directions
                rev_path = [reverse_action(a) for a in rev_path]
                best_path = path + rev_path
                mu = best_cost
        if curr not in visited_fwd or g < visited_fwd[curr]:
            visited_fwd[curr] = g
            for succ, act, cost in problem.get_successors(curr):
                new_path = path + [act]
                new_cost = problem.get_cost_of_actions(new_path)
                frontier_fwd.push((succ, new_path), new_cost + heuristic(succ, problem))
                if succ not in path_fwd or len(new_path) < len(path_fwd[succ]):
                    path_fwd[succ] = new_path

        # Expand backward frontier
        curr_b, path_b = frontier_bwd.pop()
        g_b = problem.get_cost_of_actions(path_b)
        if curr_b in visited_fwd:
            total_cost = g_b + visited_fwd[curr_b]
            if total_cost < best_cost:
                best_cost = total_cost
                rev_path = path_fwd[curr_b][::-1]
                rev_path = [reverse_action(a) for a in rev_path]
                best_path = path_b + rev_path
                mu = best_cost
        if curr_b not in visited_bwd or g_b < visited_bwd[curr_b]:
            visited_bwd[curr_b] = g_b
            for succ, act, cost in problem.get_successors(curr_b):
                rev_act = reverse_action(act)
                new_path = path_b + [rev_act]
                new_cost = problem.get_cost_of_actions(new_path)
                frontier_bwd.push((succ, new_path), new_cost + heuristic(succ, problem))
                if succ not in path_bwd or len(new_path) < len(path_bwd[succ]):
                    path_bwd[succ] = new_path
    return best_path

def reverse_action(action):
    reverse = {'North': 'South', 'South': 'North', 'East': 'West', 'West': 'East'}
    return reverse.get(action, action)

class AlphaBetaAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        def alpha_beta(state, depth, agent_index, alpha, beta):
            if state.is_win() or state.is_lose() or depth == self.depth * state.get_num_agents():
                return self.evaluation_function(state)
            if agent_index == 0:  # Pacman max node
                value = float('-inf')
                for action in state.get_legal_actions(agent_index):
                    successor = state.generate_successor(agent_index, action)
                    value = max(value, alpha_beta(successor, depth + 1, (agent_index + 1) % state.get_num_agents(), alpha, beta))
                    if value >= beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts min node
                value = float('inf')
                for action in state.get_legal_actions(agent_index):
                    successor = state.generate_successor(agent_index, action)
                    value = min(value, alpha_beta(successor, depth + 1, (agent_index + 1) % state.get_num_agents(), alpha, beta))
                    if value <= alpha:
                        return value
                    beta = min(beta, value)
                return value

        best_score = float('-inf')
        best_action = None
        for action in game_state.get_legal_actions(0):
            score = alpha_beta(game_state.generate_successor(0, action), 1, 1 % game_state.get_num_agents(),
                               float('-inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

def better_evaluation_function(current_game_state):
    import util
    pos = current_game_state.get_pacman_position()
    food = current_game_state.get_food()
    ghosts = current_game_state.get_ghost_states()
    capsules = current_game_state.get_capsules()
    scared_times = [ghost.scared_timer for ghost in ghosts]
    score = current_game_state.get_score()

    food_list = food.as_list()
    if food_list:
        min_food_dist = min(util.manhattan_distance(pos, food) for food in food_list)
        score += 1.0 / (min_food_dist + 1)

    score -= 4 * len(food_list)
    score -= 20 * len(capsules)

    for idx, ghost in enumerate(ghosts):
        ghost_pos = ghost.get_position()
        dist = util.manhattan_distance(pos, ghost_pos)
        if scared_times[idx] > 0:
            score += 200.0 / (dist + 1)
        else:
            if dist < 3:
                score -= 500

    return score

# Alias
better = better_evaluation_function
