# classic_game_rules.py
# ---------------------
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


from util import manhattan_distance
from game import Directions
import random, util
from game import Game
from game import GameState



class ClassicGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game ends.
    """
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.quiet = False
        self.initial_num_food = 0

    def new_game(self, layout, pacman_agent, ghost_agents, display, quiet=False):
        agents = [pacman_agent] + ghost_agents[:layout.get_num_ghosts()]
        game = Game(agents, display, self, starting_agent_index=0, mute_agents=quiet)
        game.state = GameState()
        game.state.initialize(layout, len(ghost_agents))
        game.state.data.score = 0
        self.initial_num_food = game.state.get_num_food()
        self.quiet = quiet
        return game

    def process(self, state, game, agent_index):
        """
        Checks to see whether it is time to end the game.
        """
        if state.is_win():
            self.win(state, game)
        if state.is_lose():
            self.lose(state, game)

    def win(self, state, game):
        if not self.quiet: print("Pacman emerges victorious! Score: %d" % state.data.score)
        game.game_over = True

    def lose(self, state, game):
        if not self.quiet: print("Pacman died! Score: %d" % state.data.score)
        game.game_over = True

    def get_progress(self, game):
        if self.initial_num_food == 0:
            return 1.0
        return float(game.state.get_num_food()) / self.initial_num_food

    def agent_crash(self, game, agent_index):
        if agent_index == 0:
            print("Pacman crashed")
        else:
            print("A ghost crashed")

    def get_move_timeout(self, agent_index):
        return self.timeout

