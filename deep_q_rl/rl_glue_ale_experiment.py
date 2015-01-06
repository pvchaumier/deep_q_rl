#!/usr/bin/env python
"""
This is an RLGlue experiment designed to collect the type of data
presented in:

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

(Based on the sample_experiment.py from the Rl-glue python codec examples.)

usage: rl_glue_ale_experiment.py [-h] [--num_epochs NUM_EPOCHS]
                                 [--epoch_length EPOCH_LENGTH]
                                 [--test_length TEST_LENGTH]

Author: Nathan Sprague

"""
import sys, argparse
import logging

import rlglue.RLGlue as RLGlue

def run_epoch(epoch, num_steps, prefix):
    """ Run one 'epoch' of training or testing, where an epoch is defined
    by the number of steps executed.  Prints a progress report after
    every trial

    Arguments:
       num_steps - steps per epoch
       prefix - string to print ('training' or 'testing')

    """
    steps_left = num_steps
    while steps_left > 0:
        logging.info("%s epoch: %s steps_left: %s" % (prefix, epoch, steps_left))
        terminal = RLGlue.RL_episode(steps_left)
        if not terminal:
            # if we hadn't finished yet
            RLGlue.RL_agent_message("episode_end")
        steps_left -= RLGlue.RL_num_steps()


def main(args):
    """
    Run the desired number of training epochs, a testing epoch
    is conducted after each training epoch.
    """
    from logutils import setupLogging

    parser = argparse.ArgumentParser(description='Neural rl experiment.')
    parser.add_argument("-v", "--verbose", dest="verbosity", default=0, action="count",
                      help="Verbosity.  Invoke many times for higher verbosity")

    parser.add_argument('-e', '--epochs', dest="epochs", type=int, default=100,
                        help='Number of training epochs (default: %(default)s)')
    parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch", type=int, default=50000,
                        help='Number of steps per epoch (default: %(default)s)')
    parser.add_argument('-t', '--test-length', dest="test_steps", type=int, default=10000,
                        help='Number of steps per test (default: %(default)s)')
    parameters = parser.parse_args(args)

    setupLogging(parameters.verbosity)

    RLGlue.RL_init()

    for epoch in xrange(1, parameters.epochs + 1):
        RLGlue.RL_agent_message("start_epoch " + str(epoch))
        run_epoch(epoch, parameters.steps_per_epoch, "training")
        RLGlue.RL_agent_message("finish_epoch " + str(epoch))

        if parameters.test_steps > 0:
            RLGlue.RL_agent_message("start_testing")
            run_epoch(epoch, parameters.test_steps, "testing")
            RLGlue.RL_agent_message("finish_testing " + str(epoch))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
