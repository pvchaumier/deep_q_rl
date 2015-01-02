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

from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.utils import TaskSpecVRLGLUE3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import cnn_q_learner
import ale_data_set
import theano

floatX = theano.config.floatX

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 210

CROPPED_SIZE = 80


class NeuralAgent(Agent):
    randGenerator=random.Random()

    DefaultLearningRate = 0.0001
    DefaultDiscountRate = 0.9
    DefaultEpsilonStart = 1.0
    DefaultEpsilonMin = 0.1
    DefaultEpsilonDecay = 1000000
    DefaultTestingEpsilon = 0.05
    DefaultHistoryLength = 4
    DefaultHistoryMax = 1000000
    DefaultBatchSize = 32
    DefaultPauseTime = 0

    def __init__(self, learning_rate=DefaultLearningRate,
        batch_size=DefaultBatchSize,
        discount_rate=DefaultDiscountRate,
        experiment_prefix='',
        nn_file=None,
        pause=DefaultPauseTime,
        epsilon_start=DefaultEpsilonStart,
        epsilon_min=DefaultEpsilonMin,
        epsilon_decay=DefaultEpsilonDecay,
        testing_epsilon=DefaultTestingEpsilon,
        history_length=DefaultHistoryLength,
        max_history=DefaultHistoryMax):


        self.learning_rate=learning_rate
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


        # CREATE A FOLDER TO HOLD RESULTS
        time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        if self.experiment_prefix:
            experiment_prefix = "%s_" % self.experiment_prefix
        else:
            experiment_prefix = self.experiment_prefix
        self.experiment_directory = "%s%s_%s_%s" % (experiment_prefix, time_str, str(self.learning_rate).replace(".", "p"), str(self.discount).replace(".", "p"))

        logging.info("Experiment directory: %s" % self.experiment_directory)

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
                                         batch_size=self.batch_size,
                                         approximator='cuda_conv')



    def _open_results_file(self):
        results_filename = os.path.join(self.experiment_directory, 'results.csv')
        logging.info("OPENING %s" % results_filename)
        self.results_file = open(results_filename, 'w')
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch,best_reward,mean_q,mean_q_considered\n')

    def _open_learning_file(self):
        learning_filename = os.path.join(self.experiment_directory, 'learning.csv')
        self.learning_file = open(learning_filename, 'w')
        self.learning_file.write('mean_loss,epsilon\n')

    def _update_results_file(self, epoch, num_episodes, holdout_sum):
        out = "{},{},{},{},{},{},{}\n".format(epoch, num_episodes, self.total_reward,
                                  self.total_reward / float(num_episodes), self.best_epoch_reward,
                                  holdout_sum, float(self.episode_q) / self.episode_chosen_steps)
        self.results_file.write(out)
        self.results_file.flush()


    def _update_learning_file(self):
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
            self.episode_images = [raw_image]

        return return_action


    def preprocess_observation(self, observation):
        """
        Return a processed version of the observation, and an image of the observation for record-keeping
        """

        # reshape linear to original image size
        image = observation.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        # convert from int32s
        image = np.array(image, dtype='uint8')

        # convert to an image
        pilled = Image.fromarray(image,'RGB')

        # resize and crop to a square
        if IMAGE_HEIGHT > IMAGE_WIDTH:
            resize_width = CROPPED_SIZE
            resize_height = int(round(float(IMAGE_HEIGHT) * CROPPED_SIZE / IMAGE_WIDTH))
        else:
            resize_height = CROPPED_SIZE
            resize_width = int(round(float(IMAGE_WIDTH) * CROPPED_SIZE / IMAGE_HEIGHT))

        resized = pilled.resize((resize_width, resize_height), Image.BILINEAR)

        # We select the bottom part since that's where the action happens
        # This makes a mockery of the bit above that is meant to be all general'n'shit
        crop_y = resize_height - CROPPED_SIZE
        cropped = resized.crop((0, crop_y, resize_width, resize_height))
        greyscaled = cropped.convert('L')

        arrayed = np.asarray(greyscaled)

        return arrayed, pilled


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

        #TESTING---------------------------
        if self.testing:
            self.episode_images.append(raw_image)
            self.episode_reward += reward
            int_action, max_q = self.choose_action(self.test_data_set, self.testing_epsilon,
                                             current_image, np.clip(reward, -1, 1))
            if max_q is not None:
                self.episode_chosen_steps += 1
                self.episode_q += max_q

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


    def agent_end(self, reward):
        """
        This function is called once at the end of an episode.

        Arguments:
           reward      - Real valued reward.

        Returns:
            None
        """

        logging.info("agent end being called")
        self.episode_counter += 1
        self.step_counter += 1
        total_time = time.time() - self.start_time

        if self.testing:
            self.episode_reward += reward
            if self.best_epoch_reward is None or self.episode_reward > self.best_epoch_reward:
                self.best_epoch_reward = self.episode_reward
                self.best_run_images = self.episode_images
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


    def agent_cleanup(self):
        """
        Called once at the end of an experiment.  We could save results
        here, but we use the agent_message mechanism instead so that
        a file name can be provided by the experiment.
        """

        if self.learning_file:
            self.learning_file.close()
        if self.results_file:
            self.results_file.close()

    def agent_message(self, in_message):
        """
        The experiment will cause this method to be called.  Used
        to save data to the indicated file.
        """

        logging.info("Received %s" % in_message)

        #WE NEED TO DO THIS BECAUSE agent_end is not called
        # we run out of steps.
        if in_message.startswith("episode_end"):
            self.agent_end(0)

        elif in_message.startswith("finish_epoch"):
            epoch = int(in_message.split(" ")[1])
            network_filename = os.path.join(self.experiment_directory, 'network_file_%s.pkl' % epoch)
            #net_file = open(network_filename, 'w')
            #cPickle.dump(self.network, net_file, -1)
            #net_file.close()

        elif in_message.startswith("start_testing"):
            self.testing = True
            self.total_reward = 0
            self.episode_counter = 0
            self.best_epoch_reward = None

        elif in_message.startswith("finish_testing"):
            self.testing = False
            holdout_size = 100
            epoch = int(in_message.split(" ")[1])

            if self.holdout_data is None:
                self.holdout_data = self.data_set.random_batch(holdout_size *
                                                          self.batch_size)[0]

            holdout_sum = 0
            for i in range(holdout_size * self.batch_size):
                holdout_sum += np.max(
                    self.network.q_vals(self.holdout_data[i, ...]))

            self._update_results_file(epoch, self.episode_counter,
                                      holdout_sum / (holdout_size * self.batch_size))
            self.record_best_run(epoch)
        else:
            return "I don't know how to respond to your message"


    def record_best_run(self, epoch):
        recording_directory = os.path.join(self.experiment_directory, "bestof%s_%s" % (epoch, self.best_epoch_reward))
        os.mkdir(recording_directory)

        for index, image in enumerate(self.best_run_images):
            full_name = os.path.join(recording_directory, "frame%06d.png" % index)
            image.save(full_name)

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

    parser.add_argument("-lr", '--learning-rate', dest="learning_rate", type=float,
        default=NeuralAgent.DefaultLearningRate,
        help='Learning rate (default: %(default)s)')
    parser.add_argument("-d", '--discount', dest="discount_rate", type=float, default=NeuralAgent.DefaultDiscountRate,
                        help='Discount rate (default: %(default)s)')
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

    # ignore unknowns
    parameters, _ = parser.parse_known_args(args)

    setupLogging(parameters.verbosity)

    AgentLoader.loadAgent(NeuralAgent(learning_rate=parameters.learning_rate,
        batch_size=parameters.batch_size,
        discount_rate=parameters.discount_rate,
        experiment_prefix=parameters.experiment_prefix,
        nn_file=parameters.nn_file,
        pause=parameters.pause,
        epsilon_start=parameters.epsilon_start,
        epsilon_min=parameters.epsilon_min,
        epsilon_decay=parameters.epsilon_decay,
        history_length=parameters.history_length,
        max_history=parameters.max_history))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
