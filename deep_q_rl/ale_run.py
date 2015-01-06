"""This script launches all of the processes necessary to train a
deep Q-network on an ALE game.

All unrecognized command line arguments will be passed on to
rl_glue_ale_agent.py
"""
import subprocess
import sys
import os
import argparse

DefaultBaseROMPath = "/usr/src/machinelearning/Arcade-Learning-Environment/roms/"
DefaultROM = 'breakout.bin'
DefaultPort = 4096
DefaultStepsPerEpoch = 50000
DefaultEpochs = 100
DefaultStepsPerTest = 10000
DefaultFrameSkip = 4

def main(args):
    # Check for glue_port command line argument and set it up...
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--rom', dest="rom", default=DefaultROM,
                        help='ROM to run (default: %(default)s)')
    parser.add_argument('-e', '--epochs', dest="epochs", type=int, default=DefaultEpochs,
                        help='Number of training epochs (default: %(default)s)')
    parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch", type=int, default=DefaultStepsPerEpoch,
                        help='Number of steps per epoch (default: %(default)s)')
    parser.add_argument('-t', '--test-length', dest="test_steps", type=int, default=DefaultStepsPerTest,
                        help='Number of steps per test (default: %(default)s)')    
    parser.add_argument('--no-merge', dest="merge_frames", default=True, action="store_false",
                        help='Tell ale not to send the averaged frames')    
    parser.add_argument('--experiment-prefix', dest="experiment_prefix", default=None,
        help='Experiment name prefix (default is the name of the game)')    
    parser.add_argument('--frame-skip', dest="frame_skip", default=DefaultFrameSkip, type=int,
                        help='Every how many frames to process (default: %(default)s)')        
    parser.add_argument('--glue-port', dest="glue_port", type=int, default=DefaultPort,
                        help='rlglue port (default: %(default)s)')
    parameters, unknown = parser.parse_known_args(args)

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

    if parameters.experiment_prefix is not None:
        prefix = parameters.experiment_prefix
    else:
        prefix = game_name

    # Start the necessary processes:
    p1 = subprocess.Popen(['rl_glue'], env=my_env, close_fds=close_fds)
    command = ['ale', '-game_controller', 'rlglue', '-frame_skip', str(parameters.frame_skip)]
    if not parameters.merge_frames:
        command.extend(['-disable_color_averaging', 'true'])
    command.append(full_rom_path)
    p2 = subprocess.Popen(command, env=my_env, close_fds=close_fds)
    p3 = subprocess.Popen(['./rl_glue_ale_experiment.py', '-vv', '--steps-per-epoch', str(parameters.steps_per_epoch), 
        '--test-length', str(parameters.test_steps), '--epochs', str(parameters.epochs)], env=my_env, close_fds=close_fds)
    p4 = subprocess.Popen(['./rl_glue_ale_agent.py', '-vv', '--experiment-prefix', prefix] + unknown, env=my_env, close_fds=close_fds)

    p1.wait()
    p2.wait()
    p3.wait()
    p4.wait()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
