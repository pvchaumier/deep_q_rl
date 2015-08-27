#!/usr/bin/env python
""" This script runs a pre-trained network with the game
visualization turned on.

Usage:

ale_run_watch.py NETWORK_PKL_FILE [ ROM ]
"""
import subprocess
import sys
import argparse

DefaultROM = 'breakout'
DefaultTestLength = 10000

def run_watch(args):

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-r', '--rom', dest="rom", default=DefaultROM,
                        help='ROM to run (default: %(default)s)')
    parser.add_argument('-t', '--test-length', dest="steps_per_test",
                        type=int, default=DefaultTestLength,
                        help='Number of steps per test (default: %(default)s)')
    parser.add_argument('networkfile', nargs=1,
                        help='Network file')
    parameters = parser.parse_args(args)


    command = ['./run_nature.py', '--steps-per-epoch', '0',
               '--test-length', str(parameters.steps_per_test), '--nn-file', parameters.networkfile[0],
               '--display-screen']

    if len(args) > 1:
        command.extend(['--rom', parameters.rom])

    p1 = subprocess.Popen(command)
    
    p1.wait()

    return 0

if __name__ == "__main__":
    sys.exit(run_watch(sys.argv[1:]))
