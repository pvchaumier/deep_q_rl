#! /usr/bin/env python
"""This script launches all of the processes necessary to train a
deep Q-network on an ALE game.

All unrecognized command line arguments will be passed on to
rl_glue_ale_agent.py
"""
import subprocess
import sys
import os
import logging

from parsers import OtherScriptHelper
from rl_glue_ale_agent import addScriptArguments


DefaultBaseROMPath = "../roms/"
DefaultROM = 'breakout.bin'
DefaultPort = 4096
DefaultStepsPerEpoch = 50000
DefaultEpochs = 300
DefaultStepsPerTest = 10000
DefaultFrameSkip = 4

def createParser(parser=None):
    if parser is None:
        parser = OtherScriptHelper(description=__doc__)    

    parser.add_argument("-v", "--verbose", dest="verbosity", default=0, action="count",
                        help="Verbosity.  Invoke many times for higher verbosity")
    parser.add_argument('-r', '--rom', dest="rom", default=DefaultROM,
                        help='ROM to run (default: %(default)s)')
    parser.add_argument('-e', '--epochs', dest="epochs", type=int, default=DefaultEpochs,
                        help='Number of training epochs (default: %(default)s)')
    parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch", type=int, default=DefaultStepsPerEpoch,
                        help='Number of steps per epoch (default: %(default)s)')
    parser.add_argument('-t', '--test-length', dest="test_steps", type=int, default=DefaultStepsPerTest,
                        help='Number of steps per test (default: %(default)s)')    
    parser.add_argument('--merge', dest="merge_frames", default=False, action="store_true",
                        help='Tell ALE to send the averaged frames')    
    parser.add_argument('--frame-skip', dest="frame_skip", default=DefaultFrameSkip, type=int,
                        help='Every how many frames to process (default: %(default)s)')        
    parser.add_argument('--display-screen', dest="display_screen", 
                        action='store_true', default=False,
                        help='Show the game screen.')
    parser.add_argument('--glue-port', dest="glue_port", type=int, default=DefaultPort,
                        help='rlglue port (default: %(default)s)')

    parser.other_parsers.append(addScriptArguments(None, in_group=True))

    return parser    


def run_experiment(epochs, steps_per_epoch, steps_per_test):
    from rl_glue_ale_experiment import AleExperiment

    experiment = AleExperiment(epochs, steps_per_epoch, steps_per_test)
    return experiment.run()

def run(parameters, unknown):
    from logutils import setupLogging
    setupLogging(parameters.verbosity)

    my_env = os.environ.copy()
    my_env["RLGLUE_PORT"] = str(parameters.glue_port)

    close_fds = True

    if parameters.rom.endswith('.bin'):
        rom = parameters.rom
        game_name = parameters.rom[:-4]
    else:
        game_name = parameters.rom
        rom = "%s.bin" % parameters.rom
    full_rom_path = os.path.join(DefaultBaseROMPath, rom)


    # Start the necessary processes:
    logging.info("Starting rl_glue")
    rlglue_process = subprocess.Popen(['rl_glue'], env=my_env, close_fds=close_fds)
    command = ['ale', '-game_controller', 'rlglue', '-send_rgb', 'true','-restricted_action_set', 'true', '-frame_skip', str(parameters.frame_skip)]
    if not parameters.merge_frames:
        command.extend(['-disable_color_averaging', 'true'])
    if parameters.display_screen:
        command.extend(['-display_screen', 'true'])        
    command.append(full_rom_path)
    logging.info('Starting ale: %s' % ' '.join(command))
    ale_process = subprocess.Popen(command, env=my_env, close_fds=close_fds)
    command = ['./rl_glue_ale_agent.py', '-vv', '--game-name', game_name]
    logging.info('Starting agent: %s' % ' '.join(command + unknown))
    agent_process = subprocess.Popen(command + unknown, env=my_env, close_fds=close_fds)

    logging.info("Starting experiment")
    run_experiment(parameters.epochs, parameters.steps_per_epoch, parameters.test_steps)

    rlglue_process.wait()
    ale_process.wait()
    agent_process.wait()

    return 0    


def main(args):
    parser = createParser()

    parameters, unknown = parser.parse_known_args(args)

    return run(parameters, unknown)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
