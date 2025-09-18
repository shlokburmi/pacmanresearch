from __future__ import print_function

import os
import util

class SearchProblem:
    def get_start_state(self):
        raise NotImplementedError()

    def is_goal_state(self, state):
        raise NotImplementedError()

    def get_successors(self, state):
        raise NotImplementedError()

    def get_cost_of_actions(self, actions):
        raise NotImplementedError()


# Helper: null heuristic
def null_heuristic(state, problem=None):
    return 0

# Basic search algorithms

def tiny_maze_search(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depth_first_search(problem):
    from util import Stack
    
    stack = Stack()
    visited = set()
    start_state = problem.get_start_state()
    stack.push((start_state, []))
    
    while not stack.is_empty():
        state, path = stack.pop()
        if problem.is_goal_state(state):
            return path
        if state not in visited:
            visited.add(state)
            for successor in problem.get_successors(state):
                if successor.state not in visited:
                    new_path = path + [successor.action]
                    stack.push((successor.state, new_path))
    return []

def breadth_first_search(problem):
    from util import Queue
    
    queue = Queue()
    start_state = problem.get_start_state()
    queue.push((start_state, []))
    visited = {start_state}
    
    while not queue.is_empty():
        state, path = queue.pop()
        if problem.is_goal_state(state):
            return path
        for successor in problem.get_successors(state):
            if successor.state not in visited:
                visited.add(successor.state)
                new_path = path + [successor.action]
                queue.push((successor.state, new_path))
    return []

def uniform_cost_search(problem):
    from util import PriorityQueue
    
    fringe = PriorityQueue()
    start_state = problem.get_start_state()
    fringe.push((start_state, []), 0)
    visited = {}
    
    while not fringe.is_empty():
        state, path = fringe.pop()
        if problem.is_goal_state(state):
            return path
        cost = problem.get_cost_of_actions(path)
        if state not in visited or cost < visited[state]:
            visited[state] = cost
            for successor in problem.get_successors(state):
                new_path = path + [successor.action]
                fringe.push((successor.state, new_path), problem.get_cost_of_actions(new_path))
    return []

def a_star_search(problem, heuristic=null_heuristic):
    from util import PriorityQueue
    
    fringe = PriorityQueue()
    start_state = problem.get_start_state()
    fringe.push((start_state, []), heuristic(start_state, problem))
    visited = {}
    
    while not fringe.is_empty():
        state, path = fringe.pop()
        if problem.is_goal_state(state):
            return path
        cost_so_far = problem.get_cost_of_actions(path)
        if state not in visited or cost_so_far < visited[state]:
            visited[state] = cost_so_far
            for successor in problem.get_successors(state):
                new_path = path + [successor.action]
                new_cost = problem.get_cost_of_actions(new_path) + heuristic(successor.state, problem)
                fringe.push((successor.state, new_path), new_cost)
    return []

def bidirectional_astar_search(problem, heuristic=null_heuristic):
    from util import PriorityQueue, Queue
    
    start = problem.get_start_state()
    if problem.is_goal_state(start):
        return []
    
    goal = getattr(problem, 'goal', None)
    if goal is None:
        q = Queue()
        q.push(start)
        visited_goal_search = {start}
        while not q.is_empty():
            state = q.pop()
            if problem.is_goal_state(state):
                goal = state
                break
            for successor in problem.get_successors(state):
                if successor.state not in visited_goal_search:
                    visited_goal_search.add(successor.state)
                    q.push(successor.state)
        if goal is None:
            return a_star_search(problem, heuristic)
    
    forward_fringe = PriorityQueue()
    backward_fringe = PriorityQueue()
    forward_fringe.push((start, []), heuristic(start, problem))
    backward_fringe.push((goal, []), heuristic(goal, problem))
    forward_visited = {start: (0, [])}
    backward_visited = {goal: (0, [])}
    
    best_cost = float('inf')
    best_path = []
    
    while not forward_fringe.is_empty() and not backward_fringe.is_empty():
        if not forward_fringe.is_empty():
            curr_state, path = forward_fringe.pop()
            cost_so_far = problem.get_cost_of_actions(path)
            if curr_state in backward_visited:
                b_cost, b_path = backward_visited[curr_state]
                total_cost = cost_so_far + b_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = path + _reverse_path(b_path)
            if curr_state not in forward_visited or cost_so_far < forward_visited[curr_state][0]:
                forward_visited[curr_state] = (cost_so_far, path)
                for successor in problem.get_successors(curr_state):
                    new_path = path + [successor.action]
                    total_estimated = problem.get_cost_of_actions(new_path) + heuristic(successor.state, problem)
                    forward_fringe.push((successor.state, new_path), total_estimated)
        
        if not backward_fringe.is_empty():
            curr_state, path = backward_fringe.pop()
            cost_so_far = problem.get_cost_of_actions(path)
            if curr_state in forward_visited:
                f_cost, f_path = forward_visited[curr_state]
                total_cost = cost_so_far + f_cost
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = f_path + _reverse_path(path)
            if curr_state not in backward_visited or cost_so_far < backward_visited[curr_state][0]:
                backward_visited[curr_state] = (cost_so_far, path)
                for successor in problem.get_successors(curr_state):
                    rev_action = _reverse_action(successor.action)
                    new_path = path + [rev_action]
                    total_estimated = problem.get_cost_of_actions(new_path) + heuristic(successor.state, problem)
                    backward_fringe.push((successor.state, new_path), total_estimated)
    
    return best_path

def _reverse_action(action):
    reverse_map = {
        "North": "South",
        "South": "North",
        "East": "West",
        "West": "East",
        "Stop": "Stop"
    }
    return reverse_map.get(action, action)
    
def _reverse_path(path):
    return [_reverse_action(a) for a in reversed(path)]

def bidirectional_bfs_search(problem):
    from util import Queue
    
    start = problem.get_start_state()
    if problem.is_goal_state(start):
        return []
    
    goal = getattr(problem, 'goal', None)
    if goal is None:
        q = Queue()
        q.push(start)
        visited_goal = {start}
        while not q.is_empty():
            state = q.pop()
            if problem.is_goal_state(state):
                goal = state
                break
            for suc in problem.get_successors(state):
                if suc.state not in visited_goal:
                    visited_goal.add(suc.state)
                    q.push(suc.state)
        if goal is None:
            return breadth_first_search(problem)
    
    forward_queue = Queue()
    backward_queue = Queue()
    forward_queue.push((start, []))
    backward_queue.push((goal, []))
    forward_visited = {start: []}
    backward_visited = {goal: []}
    
    while not forward_queue.is_empty() and not backward_queue.is_empty():
        if not forward_queue.is_empty():
            current, path = forward_queue.pop()
            if current in backward_visited:
                return path + _reverse_path(backward_visited[current])
            for suc in problem.get_successors(current):
                if suc.state not in forward_visited:
                    new_path = path + [suc.action]
                    forward_visited[suc.state] = new_path
                    forward_queue.push((suc.state, new_path))
        if not backward_queue.is_empty():
            current, path = backward_queue.pop()
            if current in forward_visited:
                return forward_visited[current] + _reverse_path(path)
            for suc in problem.get_successors(current):
                rev_action = _reverse_action(suc.action)
                if suc.state not in backward_visited:
                    new_path = path + [rev_action]
                    backward_visited[suc.state] = new_path
                    backward_queue.push((suc.state, new_path))
    return []

# Aliases for command line usage
bfs = breadth_first_search
dfs = depth_first_search
ucs = uniform_cost_search
astar = a_star_search
bidirectional_bfs = bidirectional_bfs_search
bidirectional_astar = bidirectional_astar_search
