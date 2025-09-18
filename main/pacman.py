# pacman.py
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
Pacman.py holds the logic for the classic Pacman game along with the main
code to run a game.  This file is divided into three sections:

  (i)  Your interface to the pacman world:
          PacmanGame and GameState

  (ii) The Pacman running code:
          Game

  (iii) Framework for implementing Pacman agents:
          Agent, Directions, and Actions

"""
import os
import sys

# --- START ROBUST PATHING FIX ---
# This ensures that the script can find all its necessary files and modules
# no matter where you run it from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
# --- END ROBUST PATHING FIX ---


from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearest_point
from util import manhattan_distance
import util, layout
import types, time, random, traceback

##################################################################
# YOUR INTERFACE TO THE PACMAN WORLD (USEFUL FOR AGENTS) #
##################################################################

class PacmanGame:
    """
    This class manages the running of a game, reading layouts, and leaving
    agents to their own devices.
    """

    def __init__(self, agents, display, rules, starting_index=0, mute_agents=False, catch_exceptions=False):
        self.agent_crashed = False
        self.agents = agents
        self.display = display
        self.rules = rules
        self.starting_index = starting_index
        self.game_over = False
        self.mute_agents = mute_agents
        self.catch_exceptions = catch_exceptions
        self.state = None

    def get_progress(self):
        if self.game_over:
            return 1.0
        else:
            return self.rules.get_progress(self.state)

    def _agent_crash(self, agent_index, quiet=False):
        "Helper method for handling agent crashes"
        if not quiet:
            traceback.print_exc()
        self.game_over = True
        self.agent_crashed = True
        self.rules.agent_crash(self.state, agent_index)

    def run(self):
        """
        Main control loop for game play.
        """
        self.display.initialize(self.state.data)
        self.num_moves = 0
        
        # inform agents of the game start
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if not agent:
                # this is a null agent, meaning it failed to load
                # the other team wins
                print("Agent %d failed to load" % i, file=sys.stderr)
                self.agent_crashed = True
                self.game_over = True
                return

            if ("register_initial_state" in dir(agent)):
                if self.catch_exceptions:
                    try:
                        agent.register_initial_state(self.state.deep_copy())
                    except Exception as data:
                        self._agent_crash(i)
                        return
                else:
                    agent.register_initial_state(self.state.deep_copy())

        agent_index = self.starting_index
        num_agents = len(self.agents)

        while not self.game_over:
            # Fetch the next agent
            agent = self.agents[agent_index]
            move_time = 0
            skip_action = False
            # Generate an action
            if self.catch_exceptions:
                try:
                    timed_action = util.timeout(agent.get_action,
                                                args=[self.state.deep_copy()],
                                                time_limit=self.rules.get_move_timeout(agent_index))
                except Exception as data:
                    self._agent_crash(agent_index)
                    return
            else:
                timed_action = agent.get_action(self.state.deep_copy())

            action = timed_action

            # Execute the action
            self.state = self.state.generate_successor(agent_index, action)
            self.rules.process(self.state, self, agent_index)
            # Change the display
            self.display.update(self.state.data)

            # Next agent
            agent_index = (agent_index + 1) % num_agents
            self.num_moves += 1
        # inform a learning agent of the game result
        for agent_index, agent in enumerate(self.agents):
            if "final" in dir(agent):
                if self.catch_exceptions:
                    try:
                        agent.final(self.state)
                    except Exception as data:
                        self._agent_crash(agent_index)
                        return
                else:
                    agent.final(self.state)
        self.display.finish()


def read_command(argv):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usage_str = """
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python pacman.py
                    - starts an interactive game
                (2) python pacman.py --layout small_classic --zoom 2
                OR  python pacman.py -l small_classic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    """
    parser = OptionParser(usage_str)

    parser.add_option('-n', '--num_games', dest='num_games', type='int',
                      help='the number of GAMES to play', metavar='GAMES', default=1)
    parser.add_option('-l', '--layout', dest='layout',
                      help='the LAYOUT_FILE from which to load the map',
                      metavar='LAYOUT_FILE', default='medium_classic')
    parser.add_option('-p', '--pacman', dest='pacman',
                      help='the agent TYPE in search_agents.py to use',
                      metavar='TYPE', default='KeyboardAgent')
    parser.add_option('-t', '--text_graphics', action='store_true', dest='text_graphics',
                      help='Display output as text only', default=False)
    parser.add_option('-q', '--quiet_graphics', action='store_true', dest='quiet_graphics',
                      help='Generate minimal output and no graphics', default=False)
    parser.add_option('-g', '--ghosts', dest='ghost',
                      help='the ghost agent TYPE in ghost_agents.py to use',
                      metavar='TYPE', default='RandomGhost')
    parser.add_option('-k', '--numghosts', type='int', dest='num_ghosts',
                      help='The number of ghosts to use', default=4)
    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                      help='Zoom the size of the graphics window', default=1.0)
    parser.add_option('-f', '--fix_random_seed', action='store_true', dest='fix_random_seed',
                      help='Fixes the random seed to always play the same game', default=False)
    parser.add_option('-r', '--record_actions', action='store_true', dest='record',
                      help='Writes game histories to a file (named by the time they were played)', default=False)
    parser.add_option('--replay', dest='game_to_replay',
                      help='A recorded game file from which to replay the game', default=None)
    parser.add_option('-a', '--agent_args', dest='agent_args',
                      help='Comma separated values sent to agent. e.g. "opt1=val1,opt2,opt3=val3"')
    parser.add_option('-x', '--num_training', dest='num_training', type='int',
                      help='How many episodes are training (suppresses output)', default=0)
    parser.add_option('--frame_time', dest='frame_time', type='float',
                      help='Time to delay between frames; <0 means keyboard', default=0.1)
    parser.add_option('-c', '--catch_exceptions', action='store_true', dest='catch_exceptions',
                      help='Turns on exception handling and timeouts during games', default=False)
    parser.add_option('--timeout', dest='timeout', type='int',
                      help='Maximum time agents can spend thinking in a single turn', default=30)
    parser.add_option('--start', dest='start_point', type='string',
                      help='Specify the start state for the search problem, e.g., "1,1"',
                      default=None)
    parser.add_option('--goal', dest='goal_point', type='string',
                      help='Specify the goal state for the search problem, e.g., "10,10"',
                      default=None)
                      
    (options, args) = parser.parse_args(argv)
    
    agent_args = {}
    if options.agent_args:
        for arg in options.agent_args.split(','):
            if '=' in arg:
                key, val = arg.split('=')
            else:
                key, val = arg, True
            agent_args[key] = val

    if options.start_point:
        try:
            startX, startY = [int(x) for x in options.start_point.split(',')]
            agent_args['start'] = (startX, startY)
        except:
            print('Invalid start_point format. Use X,Y, e.g., "1,1"', file=sys.stderr)
            sys.exit(1)

    if options.goal_point:
        try:
            goalX, goalY = [int(x) for x in options.goal_point.split(',')]
            agent_args['goal'] = (goalX, goalY)
        except:
            print('Invalid goal_point format. Use X,Y, e.g., "10,10"', file=sys.stderr)
            sys.exit(1)

    if options.quiet_graphics:
        import text_display
        display = text_display.NullGraphics()
    elif options.text_graphics:
        import text_display
        text_display.SLEEP_TIME = options.frame_time
        display = text_display.PacmanGraphics()
    else:
        import graphics_display
        display = graphics_display.PacmanGraphics(options.zoom, frame_time=options.frame_time)

    try:
        layout_obj = layout.get_layout(options.layout)
        if layout_obj == None:
            raise Exception("Layout not found.")
    except Exception as e:
        print(('Could not load layout file: %s' % options.layout), file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    no_keyboard = options.game_to_replay is None and (options.text_graphics or options.quiet_graphics)
    pacman_type = load_agent(options.pacman, no_keyboard)
    ghost_type = load_agent(options.ghost, no_keyboard)
    pacman = pacman_type(**agent_args) 
    ghosts = [ghost_type(i+1) for i in range(options.num_ghosts)]

    import classic_game_rules
    rules = classic_game_rules.ClassicGameRules(options.timeout)

    from game import GameState
    game = PacmanGame([pacman] + ghosts, display, rules, catch_exceptions=options.catch_exceptions)
    game.state = GameState()
    game.state.initialize(layout_obj, options.num_ghosts)

    return {'game': game, 'num_games': options.num_games, 'record': options.record, 'num_training': options.num_training, 'fix_random_seed': options.fix_random_seed}


def load_agent(agent_name, no_keyboard):
    try:
        module = __import__('search_agents')
        if agent_name in dir(module):
            return getattr(module, agent_name)
    except ImportError:
        pass
    
    try:
        module = __import__('ghost_agents')
        if agent_name in dir(module):
            return getattr(module, agent_name)
    except ImportError:
        pass
        
    raise Exception("The agent '" + agent_name + "' is not specified in any *Agents.py.")


def run_games(game, num_games, record, num_training, fix_random_seed):
    import __main__
    __main__.__dict__['_display'] = game.display

    if fix_random_seed:
        random.seed('cs188')
        
    scores = []
    for i in range(num_games):
        if not game.rules.quiet:
            print(('Episode %d of %d' % (i+1, num_games)))
        
        game.run()
        scores.append(game.state.get_score())
        if record:
            import pickle
            fname = ('recorded-game-%d' % (i+1)) + '-'.join([str(t) for t in time.localtime()[1:6]])
            with open(fname, 'wb') as f:
                components = {'layout': game.state.data.layout, 'agents': game.agents, 'display': game.display, 'rules': game.rules}
                pickle.dump(components, f)

    if num_games > 1:
        print(('Average Score:', sum(scores) / float(num_games)))
        print(('Scores:       ', ', '.join([str(score) for score in scores])))
    win_rate = [s > 0 for s in scores].count(True) / float(len(scores))
    print(('Win Rate:      %d/%d (%.2f)' % ([s > 0 for s in scores].count(True), len(scores), win_rate)))
    print(('Record:       ', ', '.join([('Loss', 'Win')[int(s > 0)] for s in scores])))

    return scores

if __name__ == '__main__':
    args = read_command(sys.argv[1:]) 
    run_games(**args)

