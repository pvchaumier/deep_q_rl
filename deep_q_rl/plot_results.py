"""Plots data corresponding to Figure 2 in

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

"""

import sys, os
import logging

import numpy as np
import matplotlib.pyplot as plt


DefaultTrainedEpoch = 100

def read_data(filename):
    input_file = open(filename, "rb")
    results = np.loadtxt(input_file, delimiter=",", skiprows=1)
    input_file.close()

    return results


def plot(results, plot_q_values, plot_max_values, game_name):
    # Modify this to do some smoothing...
    kernel = np.array([1.] * 1)
    kernel = kernel / np.sum(kernel)

    plot_count = 1

    if plot_q_values:
        plot_count += 1

    if plot_max_values:
        plot_count += 1

    scores = plt.subplot(1, plot_count, 1)
    plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), '-*')
    scores.set_xlabel('epoch')
    scores.set_ylabel('score')

    current_sub_plot = 2

    if plot_max_values:
        
        max_values = plt.subplot(1, plot_count, current_sub_plot)
        current_sub_plot += 1
        plt.plot(results[:, 0], results[:, 4], 'r-.')
        max_values.set_xlabel('epoch')
        max_values.set_ylabel('Max score')
        y_limits = max_values.get_ylim()
        # set main score's limits to be the same as this one to make comparison easier
        scores.set_ylim(y_limits)

    if plot_q_values:
        qvalues = plt.subplot(1, plot_count, current_sub_plot)
        current_sub_plot += 1
        plt.plot(results[:, 0], results[:, 5], '-')
        qvalues.set_xlabel('epoch')
        qvalues.set_ylabel('Q value')



    if game_name and plot_count == 1:
        scores.set_title(game_name)

    plt.show()



def main(args):

    from argparse import ArgumentParser
    from logutils import setupLogging
    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", dest="verbosity", default=0, action="count",
                      help="Verbosity.  Invoke many times for higher verbosity")    
    parser.add_argument("-g", "--game-name", dest="game_name", default=None,
        help="Name of game to put on title")
    parser.add_argument("--no-q", dest="plotQValues", default=True, action="store_false",
        help="Don't plot the Q values")    
    parser.add_argument("--no-max", dest="plotMaxValues", default=True, action="store_false",
        help="Don't plot the max values")        
    parser.add_argument("-t", "--trained-epoch", dest="trained_epoch", default=DefaultTrainedEpoch, type=int,
        help="Epoch at which we consider the network as trained (default: %(default)s)")            
    parser.add_argument("results", nargs=1,
        help="Results file")

    parameters = parser.parse_args(args)

    setupLogging(parameters.verbosity)

    results = read_data(os.path.expanduser(parameters.results[0]))
    plot(results, parameters.plotQValues, parameters.plotMaxValues, parameters.game_name)

    logging.info("Average score after %d epochs: %s" % (parameters.trained_epoch, np.mean(results[parameters.trained_epoch:, 3])))

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
