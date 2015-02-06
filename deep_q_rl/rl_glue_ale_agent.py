#!/usr/bin/env python
"""
This uses the skeleton_agent.py file from the Python-codec of rl-glue
as a starting point.


Author: Nathan Sprague
"""

#
# Copyright (C) 2008, Brian Tanner
#
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import copy
import os
import cPickle
import time
import random
import argparse
import logging
import datetime
import sys
sys.setrecursionlimit(10000)

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.utils import TaskSpecVRLGLUE3

import numpy as np
import matplotlib.pyplot as plt
import cv2
import theano

import cnn_q_learner
import ale_data_set
from playingarea import PlayingArea

floatX = theano.config.floatX

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 210

assert IMAGE_HEIGHT > IMAGE_WIDTH

CROPPED_SIZE = 84

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games. 
CROP_OFFSET = 8


class NeuralAgent(Agent):
    randGenerator=random.Random()

    DefaultLearningRate = 0.0002
    DefaultDiscountRate = 0.95
    DefaultMomentum = 0.0
    DefaultRMSDecay = 0.99
    DefaultEpsilonStart = 1.0
    DefaultEpsilonMin = 0.1
    DefaultEpsilonDecay = 1000000
    DefaultTestingEpsilon = 0.01
    DefaultHistoryLength = 4
    DefaultHistoryMax = 1000000
    DefaultBatchSize = 32
    DefaultPauseTime = 0

    def __init__(self, game_name,
        learning_rate=DefaultLearningRate,
        batch_size=DefaultBatchSize,
        discount_rate=DefaultDiscountRate,
        momentum=DefaultMomentum,
        rms_decay=DefaultRMSDecay,
        experiment_prefix='',
        nn_file=None,
        pause=DefaultPauseTime,
        epsilon_start=DefaultEpsilonStart,
        epsilon_min=DefaultEpsilonMin,
        epsilon_decay=DefaultEpsilonDecay,
        testing_epsilon=DefaultTestingEpsilon,
        history_length=DefaultHistoryLength,
        max_history=DefaultHistoryMax,
        best_video=True,
        keep_epoch_network=True,
        learning_log=True):


        self.game_name = game_name
        self.learning_rate=learning_rate
        self.momentum = momentum
        self.rms_decay = rms_decay
        self.batch_size=batch_size
        self.discount=discount_rate
        self.experiment_prefix=experiment_prefix
        self.nn_file=nn_file
        self.pause=pause
        self.epsilon_start=epsilon_start
        self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay
        self.phi_length=history_length
        self.max_history=max_history
        self.testing_epsilon = testing_epsilon
        self.best_video = best_video
        self.keep_epoch_network = keep_epoch_network
        self.learning_log = learning_log


        # We are going with a CV crop
        self.preprocess_observation = self._preprocess_observation_cropped_by_cv
        self.save_image = self._save_array

        if self.best_video or self.learning_log or self.keep_epoch_network:
            # CREATE A FOLDER TO HOLD RESULTS
            time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            prefices = []
            if self.experiment_prefix:
                prefices.append(experiment_prefix)
            if self.game_name:
                prefices.append(self.game_name)

            if prefices:
                experiment_prefix = "%s_" % "_".join(prefices)
            else:
                experiment_prefix = ''

            self.experiment_directory = "%s%s_%s_%s" % (experiment_prefix, time_str, str(self.learning_rate).replace(".", "p"), str(self.discount).replace(".", "p"))

            logging.debug("Experiment directory: %s" % self.experiment_directory)

            try:
                os.stat(self.experiment_directory)
            except (IOError, OSError):
                os.makedirs(self.experiment_directory)

        self.learning_file = self.results_file = None


    def agent_init(self, task_spec_string):
        """
        This function is called once at the beginning of an experiment.

        Arguments: task_spec_string - A string defining the task.  This string
                                      is decoded using
                                      TaskSpecVRLGLUE3.TaskSpecParser
        """
        # DO SOME SANITY CHECKING ON THE TASKSPEC
        logging.debug("Task spec: %s" % task_spec_string)
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(task_spec_string)
        if TaskSpec.valid:

            assert ((len(TaskSpec.getIntObservations()) == 0) !=
                    (len(TaskSpec.getDoubleObservations()) == 0)), \
                "expecting continous or discrete observations.  Not both."
            assert len(TaskSpec.getDoubleActions()) == 0, \
                "expecting no continuous actions"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), \
                " expecting min action to be a number not a special value"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), \
                " expecting max action to be a number not a special value"
            self.num_actions = TaskSpec.getIntActions()[0][1]+1
        else:
            logging.error("INVALID TASK SPEC")


        self.data_set = ale_data_set.DataSet(width=CROPPED_SIZE,
                                             height=CROPPED_SIZE,
                                             max_steps=self.max_history,
                                             phi_length=self.phi_length)

        # just needs to be big enough to create phi's
        self.test_data_set = ale_data_set.DataSet(width=CROPPED_SIZE,
                                                  height=CROPPED_SIZE,
                                                  max_steps=10,
                                                  phi_length=self.phi_length)
        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = (self.epsilon - self.epsilon_min) / float(self.epsilon_decay)
        else:
            self.epsilon_rate = 0


        self.testing = False

        if self.nn_file is None:
            self.network = self._init_network()
        else:
            with open(self.nn_file, 'r') as handle:
                self.network = cPickle.load(handle)

        self._open_results_file()
        self._open_learning_file()

        self.best_score_ever = None
        self.step_counter = 0
        self.episode_counter = 0
        self.episode_reward = 0
        self.batch_counter = 0

        self.holdout_data = None

        # In order to add an element to the data set we need the
        # previous state and action and the current reward.  These
        # will be used to store states and actions.
        self.last_image = None
        self.last_action = None

        # Images for keeping best-looking runs
        self.episode_images = []
        self.best_run_images = []

        self._crop_y = self.calculate_y_crop_offset()
        logging.debug("Cropping at %s" % self._crop_y)


    def calculate_y_crop_offset(self):
        """
        Calculate the y-offset where we are going to crop the image in the target size
        """

        if self.game_name:
            playing_section = PlayingArea[self.game_name]
        else:
            playing_section = 'bottom'

        pre_crop_height = int(round(float(IMAGE_HEIGHT) * CROPPED_SIZE / IMAGE_WIDTH))
        shrink_factor = float(CROPPED_SIZE) / IMAGE_WIDTH

        if playing_section == 'top':
            return 0
        elif playing_section == 'bottom':
            return pre_crop_height - CROPPED_SIZE
        elif playing_section in ['centre', 'center']:
            return (pre_crop_height - CROPPED_SIZE) // 2
        else:
            # pixel counts
            if playing_section >= 0:
                return int(round(playing_section * shrink_factor))
            else:
                return pre_crop_height + int(round(playing_section * shrink_factor)) - CROPPED_SIZE




    def _init_network(self):
        """
        A subclass may override this if a different sort
        of network is desired.
        """
        return cnn_q_learner.CNNQLearner(self.num_actions,
                                         self.phi_length,
                                         CROPPED_SIZE,
                                         CROPPED_SIZE,
                                         discount=self.discount,
                                         learning_rate=self.learning_rate,
                                         decay=self.rms_decay,
                                         momentum=self.momentum,
                                         batch_size=self.batch_size,
                                         approximator='cuda_conv')



    def _open_results_file(self):
        if self.learning_log:
            results_filename = os.path.join(self.experiment_directory, 'results.csv')
            logging.info("OPENING %s" % results_filename)
            self.results_file = open(results_filename, 'w')
            self.results_file.write(\
                'epoch,num_episodes,total_reward,reward_per_epoch,best_reward,mean_q,mean_q_considered\n')

    def _open_learning_file(self):
        if self.learning_log:
            learning_filename = os.path.join(self.experiment_directory, 'learning.csv')
            self.learning_file = open(learning_filename, 'w')
            self.learning_file.write('mean_loss,epsilon\n')

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        if self.learning_log:
            out = "{},{},{},{},{},{},{}\n".format(epoch, num_episodes, self.total_reward,
                                      self.total_reward / max(1.0, float(num_episodes)), self.best_epoch_reward,
                                      holdout_sum, self.epoch_considered_q / max(1, self.epoch_considered_steps))
            self.results_file.write(out)
            self.results_file.flush()


    def _update_learning_file(self):
        if self.learning_log:
            out = "{},{}\n".format(np.mean(self.loss_averages),
                                   self.epsilon)
            self.learning_file.write(out)
            self.learning_file.flush()


    def agent_start(self, observation):
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - An observation of type rlglue.types.Observation

        Returns:
           An action of type rlglue.types.Action
        """

        self.step_counter = 0
        self.batch_counter = 0
        self.episode_reward = 0
        self.episode_q = 0
        self.episode_chosen_steps = 0

        # We report the mean loss for every epoch.
        self.loss_averages = []

        self.start_time = time.time()
        this_int_action = self.randGenerator.randint(0, self.num_actions-1)
        return_action = Action()
        return_action.intArray = [this_int_action]

        self.last_action = this_int_action

        self.last_image, raw_image = self.preprocess_observation(observation.intArray)
        if self.testing:
            if raw_image is not None:
                self.episode_images = [raw_image]
            else:
                self.episod_images = []


        return return_action


    def _preprocess_observation_cropped_by_cv(self, observation):
        # reshape linear to original image size
        image = observation[128:].reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        # convert from int32s
        image = np.array(image, dtype="uint8")
        greyscaled = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        resize_width = CROPPED_SIZE
        resize_height = int(round(float(IMAGE_HEIGHT) * CROPPED_SIZE / IMAGE_WIDTH))

        resized = cv2.resize(greyscaled, (resize_width, resize_height),
        interpolation=cv2.INTER_LINEAR)

        # We select a square section determined by the values set in playingarea
        cropped = resized[self._crop_y:self._crop_y + CROPPED_SIZE, :]

        return cropped, image


    def _preprocess_observation_resized_by_cv(self, observation):
        image = observation[128:].reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        # convert from int32s
        floated = np.array(image, dtype=floatX)
        greyscaled = cv2.cvtColor(floated, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(greyscaled, (CROPPED_SIZE, CROPPED_SIZE),
        interpolation=cv2.INTER_LINEAR)
        uinted = np.array(resized, dtype='uint8')
        return uinted, image


    def _record_network(self, epoch):
        if self.keep_epoch_network:
            network_filename = os.path.join(self.experiment_directory, 'network_file_%s.pkl' % epoch)
            net_file = open(network_filename, 'w')
            cPickle.dump(self.network, net_file, -1)
            net_file.close()


    def agent_step(self, reward, observation):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - An observation of type rlglue.types.Observation

        Returns:
           An action of type rlglue.types.Action

        """

        self.step_counter += 1
        return_action = Action()

        current_image, raw_image = self.preprocess_observation(observation.intArray)

        # if self.step_counter % 100 == 0:
        #     plt.imshow(current_image)
        #     plt.colorbar()
        #     plt.show()
        #     time.sleep(0.4)

        #TESTING---------------------------
        if self.testing:
            if raw_image is not None:
                self.episode_images.append(raw_image)
            self.episode_reward += reward
            int_action, max_q = self.choose_action(self.test_data_set, self.testing_epsilon,
                                             current_image, np.clip(reward, -1, 1))
            if max_q is not None:
                self.epoch_considered_steps += 1
                self.epoch_considered_q += max_q

            if self.pause > 0:
                time.sleep(self.pause)

        #NOT TESTING---------------------------
        else:
            self.epsilon = max(self.epsilon_min,
                               self.epsilon - self.epsilon_rate)

            int_action, max_q = self.choose_action(self.data_set, self.epsilon,
                                             current_image, np.clip(reward, -1, 1))

            if len(self.data_set) > self.batch_size:
                loss = self.do_training()
                self.batch_counter += 1
                self.loss_averages.append(loss)

        # Map it back to ALE's actions
        return_action.intArray = [int_action]

        self.last_action = int_action
        self.last_image = current_image

        return return_action

    def choose_action(self, data_set, epsilon, current_image, reward):
        """
        Add the most recent data to the data set and choose
        an action based on the current policy.
        """

        data_set.add_sample(self.last_image,
                            self.last_action,
                            reward, False)
        if self.step_counter >= self.phi_length:
            phi = data_set.phi(current_image)
            int_action, max_q = self.network.choose_action(phi, epsilon)
        else:
            int_action = self.randGenerator.randint(0, self.num_actions - 1)
            max_q = None
        return int_action, max_q

    def do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        """
        states, actions, rewards, next_states, terminals = \
                                self.data_set.random_batch(self.batch_size)
        return self.network.train(states, actions, rewards,
                                  next_states, terminals)


    def agent_end(self, reward, epoch_end=False):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.

        Returns:
            None
        """

        if self.step_counter <= 0:
            # Corner case where we get an end due to end of epoch just after a game ended
            return

        self.episode_counter += 1
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            self.episode_reward += reward
            if self.best_epoch_reward is None or self.episode_reward > self.best_epoch_reward:
                self.best_epoch_reward = self.episode_reward
                self.best_run_images = self.episode_images

                if self.best_score_ever is None or self.episode_reward > self.best_score_ever:
                    self.best_score_ever = self.episode_reward

            self.total_reward += self.episode_reward
        else:
            logging.info("Simulated at a rate of {} frames/s ({} batches/s) \n Average loss: {}".format(\
                self.step_counter / total_time,
                self.batch_counter/total_time,
                np.mean(self.loss_averages)))

            self._update_learning_file()

            # Store the latest sample.
            self.data_set.add_sample(self.last_image,
                                     self.last_action,
                                     np.clip(reward, -1, 1),
                                     True)


    def calculate_q_for_standard_set(self):
        """
        Calculate Q-values for a set that we'll keep reusing to track progress of Q-values
        """

        holdout_size = 3200

        if self.holdout_data is None:
            self.holdout_data = self.data_set.random_batch(holdout_size)[0]

        holdout_sum = 0
        for i in range(holdout_size):
            holdout_sum += np.max(
                self.network.q_vals(self.holdout_data[i, ...]))

        return holdout_sum / holdout_size


    def agent_cleanup(self):
        """
        Called once at the end of an experiment.  We could save results
        here, but we use the agent_message mechanism instead so that
        a file name can be provided by the experiment.
        """

        logging.info("Best score: %s" % self.best_score_ever)

        if self.learning_file:
            self.learning_file.close()
        if self.results_file:
            self.results_file.close()


    def _start_epoch(self, epoch):
        pass

    def _finish_epoch(self, epoch):
        self.agent_end(0, epoch_end=True)
        self._record_network(epoch)

    def _start_testing(self):
        self.testing = True
        self.total_reward = 0
        self.episode_counter = 0
        self.best_epoch_reward = None
        self.epoch_considered_q = 0
        self.epoch_considered_steps = 0


    def _finish_testing(self, epoch):
        self.agent_end(0, epoch_end=True)
        self.testing = False
        average_q = self.calculate_q_for_standard_set()
        self._update_results_file(epoch, self.episode_counter, average_q)
        self.record_best_run(epoch)


    def agent_message(self, in_message):
        """
        The experiment will cause this method to be called.  Used
        to save data to the indicated file.
        """

        logging.debug("Received %s" % in_message)

        if in_message.startswith("start_epoch"):
            epoch = int(in_message.split(" ")[1])
            self._start_epoch(epoch)

        elif in_message.startswith("finish_epoch"):
            epoch = int(in_message.split(" ")[1])
            self._finish_epoch(epoch)

        elif in_message.startswith("start_testing"):
            self._start_testing()

        elif in_message.startswith("finish_testing"):
            epoch = int(in_message.split(" ")[1])            
            self._finish_testing(epoch)
        else:
            return "I don't know how to respond to your message"


    def record_best_run(self, epoch):
        if self.best_video:
            recording_directory = os.path.join(self.experiment_directory, "bestof%03d_%s" % (epoch, self.best_epoch_reward))
            os.mkdir(recording_directory)

            for index, image in enumerate(self.best_run_images):
                full_name = os.path.join(recording_directory, "frame%06d.png" % index)
                self.save_image(image, full_name)


    def _save_array(self, image, filename):
        # Need to swap the colour order since cv2 expects BGR
        cv2.imwrite(filename, image[:,:,::-1])

    def _show_phis(self, phi1, phi2):
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()


def main(args):
    """
    Mostly just read command line arguments here. We do this here
    instead of agent_init to make it possible to use --help from
    the command line without starting an experiment.
    """

    from logutils import setupLogging

    # Handle command line argument:
    parser = argparse.ArgumentParser(description='Neural rl agent.')
    parser.add_argument("-v", "--verbose", dest="verbosity", default=0, action="count",
                      help="Verbosity.  Invoke many times for higher verbosity")
    parser.add_argument("-g", '--game-name', dest="game_name", default=None,
        help='Name of the game')
    parser.add_argument("-lr", '--learning-rate', dest="learning_rate", type=float,
        default=NeuralAgent.DefaultLearningRate,
        help='Learning rate (default: %(default)s)')
    parser.add_argument("-d", '--discount', dest="discount_rate", type=float, default=NeuralAgent.DefaultDiscountRate,
        help='Discount rate (default: %(default)s)')
    parser.add_argument("-m", '--momentum', dest="momentum", type=float, default=NeuralAgent.DefaultMomentum,
        help='Momentum term for Nesterov momentum (default: %(default)s)')    
    parser.add_argument('-r', '--rms_decay', dest="rms_decay", type=float, default=NeuralAgent.DefaultRMSDecay, 
        help='Decay rate for rms_prop (default: %(default)s)')    
    parser.add_argument('-b', '--batch-size', dest="batch_size", type=int, default=NeuralAgent.DefaultBatchSize,
        help='Batch size (default: %(default)s)')
    parser.add_argument('-e', '--experiment-prefix', dest="experiment_prefix", type=str, default="",
        help='Experiment name prefix (default: %(default)s)')
    parser.add_argument("-n", '--nn-file', dest="nn_file", type=str, default=None,
        help='Pickle file containing trained net. (default: %(default)s)')
    parser.add_argument("-p", '--pause', dest="pause", type=float, default=NeuralAgent.DefaultPauseTime,
        help='Amount of time to pause display while testing. (default: %(default)s)')
    parser.add_argument("-es", '--epsilon-start', dest="epsilon_start", type=float,
        default=NeuralAgent.DefaultEpsilonStart,
        help='Starting value for epsilon. (default: %(default)s)')
    parser.add_argument('--epsilon-min', dest="epsilon_min", type=float, default=NeuralAgent.DefaultEpsilonMin,
        help='Minimum epsilon. (default: %(default)s)')
    parser.add_argument('--epsilon-decay', dest="epsilon_decay", type=float, default=NeuralAgent.DefaultEpsilonDecay,
        help='Number of steps to minimum epsilon. (default: %(default)s)')
    parser.add_argument("-hl", '--history-length', dest="history_length", type=int, default=NeuralAgent.DefaultHistoryLength,
        help='History length (default: %(default)s)')
    parser.add_argument('--max-history', dest="max_history", type=int, default=NeuralAgent.DefaultHistoryMax,
        help='Maximum number of steps stored (default: %(default)s)')
    parser.add_argument('--no-video', dest="video", default=True, action="store_false",
        help='Do not make a "video" record of the best run in each game')    
    parser.add_argument('--no-records', dest="recording", default=True, action="store_false",
        help='Do not record anything about the experiment (best games, epoch networks, test results, etc)')


    # ignore unknowns
    parameters, _ = parser.parse_known_args(args)

    setupLogging(parameters.verbosity)

    if not parameters.recording:
        best_video = epoch_network = learning_log = False
    else:
        best_video = parameters.video
        epoch_network = learning_log = True

    AgentLoader.loadAgent(NeuralAgent(parameters.game_name,
        learning_rate=parameters.learning_rate,
        batch_size=parameters.batch_size,
        discount_rate=parameters.discount_rate,
        experiment_prefix=parameters.experiment_prefix,
        nn_file=parameters.nn_file,
        pause=parameters.pause,
        epsilon_start=parameters.epsilon_start,
        epsilon_min=parameters.epsilon_min,
        epsilon_decay=parameters.epsilon_decay,
        history_length=parameters.history_length,
        max_history=parameters.max_history,
        best_video=best_video,
        keep_epoch_network=epoch_network,
        learning_log=learning_log))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
