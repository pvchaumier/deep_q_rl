""" This script runs a pre-trained network with the game
visualization turned on.

Usage:

ale_run_watch.py [usual parameters to ale_run.py]
"""
import sys

from ale_run import createParser, run

DefaultGluePort = 4097
DefaultTestSteps = 1000000

def run_watch(args):
    parser = createParser()

    parser.set_defaults(glue_port=DefaultGluePort, steps_per_epoch=0, display_screen=True, epochs=1, 
        test_steps=DefaultTestSteps)

    parameters, unknown = parser.parse_known_args(args)


    if '--pause' not in unknown and '-p' not in unknown:
        unknown.extend(['-p', '0.03'])

    if '--max-history' not in unknown:
        unknown.extend(['--max-history', '4000']) # small history to reduce useless training space

    return run(parameters, unknown)


if __name__ == "__main__":
    sys.exit(run_watch(sys.argv[1:]))
