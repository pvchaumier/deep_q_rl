"""Plots data corresponding to Figure 2 in

Playing Atari with Deep Reinforcement Learning
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis
Antonoglou, Daan Wierstra, Martin Riedmiller

"""

import sys, os

import numpy as np
import matplotlib.pyplot as plt


def plot(filename, game_name):
    # Modify this to do some smoothing...
    kernel = np.array([1.] * 1)
    kernel = kernel / np.sum(kernel)

    input_file = open(filename, "rb")
    results = np.loadtxt(input_file, delimiter=",", skiprows=1)
    scores = plt.subplot(1, 2, 1)
    plt.plot(results[:, 0], np.convolve(results[:, 3], kernel, mode='same'), '-*')
    scores.set_xlabel('epoch')
    scores.set_ylabel('score')


    qvalues = plt.subplot(1, 2, 2)
    plt.plot(results[:, 0], results[:, 5], '--')
    qvalues.set_xlabel('epoch')
    qvalues.set_ylabel('Q value')


    input_file.close()
    plt.show()



def main(args):

    from argparse import ArgumentParser
    
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--game-name", dest="game_name", default=None,
        help="Name of game to put on title")
    parser.add_argument("results", nargs=1,
        help="Results file")

    parameters = parser.parse_args(args)

    plot(os.path.expanduser(parameters.results[0]), parameters.game_name)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
