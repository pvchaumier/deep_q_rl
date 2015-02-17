#!/usr/bin/env python
"""
Agent that is only used for testing. It loads each of the network file descriptors in turn
and outputs the result of it playing

This is mainly used to rerun with a different testing epsilon
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



class TestingNeuralAgent(Agent):
    randGenerator=random.Random()

    DefaultTestingEpsilon = 0.01
    DefaultHistoryLength = 4
    DefaultBatchSize = 32
    DefaultPauseTime = 0

    def __init__(self, game_name,
        batch_size=DefaultBatchSize,
        experiment_directory='',
        pause=DefaultPauseTime,
        testing_epsilon=DefaultTestingEpsilon,
        history_length=DefaultHistoryLength,
        best_video=True,
        learning_log=True):


        self.game_name = game_name
        self.batch_size=batch_size
        self.experiment_directory=experiment_directory
        self.pause=pause
        self.phi_length=history_length
        self.testing_epsilon = testing_epsilon
        self.best_video = best_video
        self.learning_log = learning_log


        # We are going with a CV crop
        self.preprocess_observation = self._preprocess_observation_cropped_by_cv
        self.save_image = self._save_array

        if self.best_video or self.learning_log:
            # Create a subdirectory to hold results
            time_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            subdirectory_name = 'testing_%s_%s' % (self.testing_epsilon, time_string)
            self.output_directory = os.path.join(self.experiment_directory, subdirectory_name)

            logging.debug("Output directory: %s" % self.output_directory)

            try:
                os.stat(self.output_directory)
            except (IOError, OSError):
                os.makedirs(self.output_directory)

        self.results_file = None


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


        self.test_data_set = ale_data_set.DataSet(width=CROPPED_SIZE,
                                                  height=CROPPED_SIZE,
                                                  max_steps=10,
                                                  phi_length=self.phi_length)


        self._open_results_file()

        self.best_score_ever = None
        self.step_counter = 0
        self.episode_counter = 0
        self.episode_reward = 0
        self.batch_counter = 0
        self.epoch_counter = 0


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



    def _open_results_file(self):
        if self.learning_log:
            results_filename = os.path.join(self.output_directory, 'results.csv')
            logging.info("OPENING %s" % results_filename)
            self.results_file = open(results_filename, 'w')
            self.results_file.write(\
                'epoch,num_episodes,total_reward,reward_per_epoch,best_reward,mean_q,mean_q_considered\n')


    def _update_results_file(self, epoch, num_episodes):
        if self.learning_log:
            out = "{},{},{},{},{}\n".format(epoch, num_episodes, self.total_reward,
                                      self.total_reward / max(1.0, float(num_episodes)), self.best_epoch_reward)
            self.results_file.write(out)
            self.results_file.flush()


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
        if raw_image is not None:
            self.episode_images.append(raw_image)
        self.episode_reward += reward
        int_action, max_q = self.choose_action(self.test_data_set, self.testing_epsilon,
                                         current_image, np.clip(reward, -1, 1))
        if max_q is not None:
            self.epoch_considered_steps += 1

        if self.pause > 0:
            time.sleep(self.pause)

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


    def agent_end(self, reward, testing_stop=False):
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

        self.step_counter += 1
        total_time = time.time() - self.start_time
        logging.info("FPS in that episode: %s" % (float(self.step_counter)/ total_time))

        self.episode_reward += reward
            
        if self.best_epoch_reward is None or self.episode_reward > self.best_epoch_reward:
            self.best_epoch_reward = self.episode_reward
            self.best_run_images = self.episode_images

            if self.best_score_ever is None or self.episode_reward > self.best_score_ever:
                self.best_score_ever = self.episode_reward

        if not testing_stop or self.episode_counter == 0:
            # only collect stats for this episode if it didn't get truncated, or if it's the 
            # only one we are going to get for this testing epoch
            self.episode_counter += 1
            self.total_reward += self.episode_reward


    def agent_cleanup(self):
        """
        Called once at the end of an experiment.  We could save results
        here, but we use the agent_message mechanism instead so that
        a file name can be provided by the experiment.
        """

        logging.info("Best score: %s" % self.best_score_ever)

        if self.results_file:
            self.results_file.close()


    def _start_testing(self):
        self.total_reward = 0
        self.episode_counter = 0
        self.best_epoch_reward = None
        self.epoch_considered_steps = 0
        self.epoch_counter += 1
        # load the new network
        network_filename = os.path.join(self.experiment_directory, 'network_file_%s.pkl' % self.epoch_counter)
        with open(network_filename, 'r') as handle:
            self.network = cPickle.load(handle)




    def _finish_testing(self, epoch):
        self.agent_end(0, True)
        self._update_results_file(epoch, self.episode_counter)
        self.record_best_run(epoch)


    def agent_message(self, in_message):
        """
        The experiment will cause this method to be called.  Used
        to save data to the indicated file.
        """

        logging.debug("Received %s" % in_message)

        if in_message.startswith("start_testing"):
            self._start_testing()

        elif in_message.startswith("finish_testing"):
            epoch = int(in_message.split(" ")[1])            
            self._finish_testing(epoch)
        else:
            return "I don't know how to respond to your message"


    def record_best_run(self, epoch):
        if self.best_video:
            recording_directory = os.path.join(self.output_directory, "bestof%03d_%s" % (epoch, self.best_epoch_reward))
            os.mkdir(recording_directory)

            for index, image in enumerate(self.best_run_images):
                full_name = os.path.join(recording_directory, "frame%06d.png" % index)
                self.save_image(image, full_name)


    def _save_array(self, image, filename):
        # Need to swap the colour order since cv2 expects BGR
        cv2.imwrite(filename, image[:,:,::-1])



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
    parser.add_argument('-b', '--batch-size', dest="batch_size", type=int, default=TestingNeuralAgent.DefaultBatchSize,
        help='Batch size (default: %(default)s)')
    parser.add_argument('-e', '--experiment-directory', dest="experiment_directory", type=str, required=True,
        help='Directory where experiment details were saved')
    parser.add_argument('-t', '--test-epsilon', dest="testing_epsilon", type=float, default=TestingNeuralAgent.DefaultTestingEpsilon,
        help='Epsilon to use during testing (default: %(default)s)')    
    parser.add_argument("-p", '--pause', dest="pause", type=float, default=TestingNeuralAgent.DefaultPauseTime,
        help='Amount of time to pause display while testing. (default: %(default)s)')
    parser.add_argument("-hl", '--history-length', dest="history_length", type=int, default=TestingNeuralAgent.DefaultHistoryLength,
        help='History length (default: %(default)s)')
    parser.add_argument('--no-video', dest="video", default=True, action="store_false",
        help='Do not make a "video" record of the best run in each game')    
    parser.add_argument('--no-records', dest="recording", default=True, action="store_false",
        help='Do not record anything about the experiment (best games, epoch networks, test results, etc)')


    # ignore unknowns
    parameters, _ = parser.parse_known_args(args)

    setupLogging(parameters.verbosity)

    if not parameters.recording:
        best_video = learning_log = False
    else:
        best_video = parameters.video
        learning_log = True

    AgentLoader.loadAgent(TestingNeuralAgent(parameters.game_name,
        batch_size=parameters.batch_size,
        experiment_directory=parameters.experiment_directory,
        testing_epsilon=parameters.testing_epsilon,
        pause=parameters.pause,
        history_length=parameters.history_length,
        best_video=best_video,
        learning_log=learning_log))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
