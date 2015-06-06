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
import cv2


IMAGE_WIDTH = 160
IMAGE_HEIGHT = 210

assert IMAGE_HEIGHT > IMAGE_WIDTH
CROPPED_SIZE = 84


class KnowWhereYouComeFrom(argparse.Action):
    """
    Store action that remembers that also stores whether the value comes from being set or 
    from the default
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values) # standard
        if hasattr(namespace, 'manually_set'):
            namespace.manually_set.add(self.dest)
        else:
            namespace.manually_set = set([self.dest])


class DefaultParameters(object):
    LearningRate = 0.00025

    RmsDecay = 0.95
    EpsilonStart = 1.0
    EpsilonMin = 0.1
    EpsilonDecay = 1000000
    TestingEpsilon = 0.01
    HistoryLength = 4
    HistoryMax = 1000000
    BatchSize = 32
    PauseTime = 0
    TargetResetFrequency = 10000
    Momentum = 0.95
    NetworkSize = 'big'
    #NetworkSize = 'big_cudnn'
    DiscountRate = 0.99
    ReplayStartSize = 50000
    ImagePreparation = 'resize'

    @classmethod
    def get_default(cls, parameters, variable):
        """
        Get the default value for the variable from the default parameters
        unless it has been manually overridden
        """

        manually_set = getattr(parameters, 'manually_set', set())
        if variable in manually_set:
            return getattr(parameters, variable)

        parts = variable.split('_')
        parts = [part.capitalize() for part in parts]
        full_name = ''.join(parts)
        return getattr(cls, full_name, getattr(parameters, variable, None))


class NIPSParameters(DefaultParameters):

    LearningRate = 0.0002
    Momentum = 0.0
    NetworkSize = 'small'
    TargetResetFrequency = 0
    DiscountRate = 0.95
    ReplayStartSize = 0
    ImagePreparation = 'crop'


class NeuralAgent(Agent):
    randGenerator=random.Random()


    def __init__(self, game_name, network_size=DefaultParameters.NetworkSize,
        learning_rate=DefaultParameters.LearningRate,
        batch_size=DefaultParameters.BatchSize,
        discount_rate=DefaultParameters.DiscountRate,
        momentum=DefaultParameters.Momentum,
        rms_decay=DefaultParameters.RmsDecay,
        experiment_prefix='',
        experiment_directory=None,
        nn_file=None,
        pause=DefaultParameters.PauseTime,
        epsilon_start=DefaultParameters.EpsilonStart,
        epsilon_min=DefaultParameters.EpsilonMin,
        epsilon_decay=DefaultParameters.EpsilonDecay,
        testing_epsilon=DefaultParameters.TestingEpsilon,
        history_length=DefaultParameters.HistoryLength,
        max_history=DefaultParameters.HistoryMax,
        replay_start_size=DefaultParameters.ReplayStartSize,
        best_video=True,
        every_video=False,
        inner_video=False,
        keep_epoch_network=True,
        learning_log=True,
        target_reset_frequency=DefaultParameters.TargetResetFrequency,
        image_preparation=DefaultParameters.ImagePreparation):


        self.game_name = game_name
        self.network_size = network_size
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
        self.inner_video = inner_video
        self.every_video = every_video
        self.keep_epoch_network = keep_epoch_network
        self.learning_log = learning_log
        self.target_reset_frequency = target_reset_frequency
        self.replay_start_size = max(replay_start_size, self.batch_size)
        self.image_preparation = image_preparation

        if image_preparation == 'crop':
            self.preprocess_observation = self._preprocess_observation_cropped_by_cv
        elif image_preparation == 'resize':
            self.preprocess_observation = self._preprocess_observation_resized_by_cv
        else:
            raise ValueError("Dunno nothing about this image preparation type: %s" % image_preparation)

        self.save_image = self._save_array

        if self.best_video or self.inner_video or self.learning_log or self.keep_epoch_network:

            if experiment_directory:
                self.experiment_directory = experiment_directory
            else:        
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
            self.record_parameters()

        self.learning_file = self.results_file = None


    def record_parameters(self):
        import subprocess

        parameters_filename = os.path.join(self.experiment_directory, 'parameters.txt')
        with open(parameters_filename, 'w') as parameters_file:
            # write the commit we are at
            gitlog = subprocess.check_output('git log -n 1 --oneline'.split()).strip()
            parameters_file.write('Last commit: %s\n' % gitlog)

            for variable in 'game_name network_size learning_rate momentum rms_decay batch_size discount nn_file pause epsilon_start epsilon_min epsilon_decay phi_length max_history testing_epsilon target_reset_frequency replay_start_size image_preparation'.split():
                parameters_file.write('%s: %s\n' % (variable, getattr(self, variable)))


        gitdiff = subprocess.check_output('git diff'.split()).strip()
        if gitdiff:
            diff_filename = os.path.join(self.experiment_directory, 'difftogit.txt')
            with open(diff_filename, 'w') as diff_file:
                diff_file.write(gitdiff)
                diff_file.write('\n')


    def agent_init(self, task_spec_string):
        """
        This function is called once at the beginning of an experiment.

        Arguments: task_spec_string - A string defining the task.  This string
                                      is decoded using
                                      TaskSpecVRLGLUE3.TaskSpecParser
        """
        import ale_data_set

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
        self.current_epoch = 0

        self.total_steps = 0
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
        self.episode_inner_images = []
        self.best_run_inner_images = []

        if self.image_preparation == 'crop':
            self._crop_y = self.calculate_y_crop_offset()
            logging.debug("Cropping at %s" % self._crop_y)


    def calculate_y_crop_offset(self):
        """
        Calculate the y-offset where we are going to crop the image in the target size
        """
        from playingarea import PlayingArea

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

        from q_network import DeepQLearner
        #from network import DeepQLearner

        if self.network_size == 'big':
             network_type = 'nature_dnn'
        else:
             network_type = 'nips_dnn'


        return DeepQLearner(CROPPED_SIZE, 
                            CROPPED_SIZE, 
                            self.num_actions, 
                            self.phi_length, 
                            batch_size=self.batch_size,
                            discount=self.discount,
                            learning_rate=self.learning_rate,
                            rho=self.rms_decay,
                            #decay=self.rms_decay,
                            momentum=self.momentum,
                            #size=self.network_size,
                            network_type=network_type,
                            #separate_evaluator=(self.target_reset_frequency > 0)
                            freeze_interval=self.target_reset_frequency
                            )


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
            self.episode_inner_images = [self.last_image]
            if raw_image is not None:
                self.episode_images = [raw_image]
            else:
                self.episode_images = []

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
        uinted = np.array(image, dtype='uint8')

        greyscaled = cv2.cvtColor(uinted, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(greyscaled, (CROPPED_SIZE, CROPPED_SIZE),
        interpolation=cv2.INTER_LINEAR)
        return resized, uinted


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
        self.total_steps += 1
        return_action = Action()

        current_image, raw_image = self.preprocess_observation(observation.intArray)

        # import matplotlib.pyplot as plt
        # if self.step_counter % 100 == 0:
        #     plt.imshow(current_image)
        #     plt.colorbar()
        #     plt.show()
        #     time.sleep(0.4)


        #TESTING---------------------------
        if self.testing:
            self.episode_inner_images.append(current_image)
            if raw_image is not None:
                self.episode_images.append(raw_image)
            self.episode_reward += reward
            int_action, considered = self.choose_action(self.test_data_set, self.testing_epsilon,
                                             current_image, np.clip(reward, -1, 1))
            if considered:
                self.epoch_considered_steps += 1

            if self.pause > 0:
                time.sleep(self.pause)

        #NOT TESTING---------------------------
        elif len(self.data_set) > self.replay_start_size:
            int_action, considered = self.choose_action(self.data_set, self.epsilon,
                                 current_image, np.clip(reward, -1, 1))

            self.epsilon = max(self.epsilon_min,
                           self.epsilon - self.epsilon_rate)
            loss = self.do_training()
            self.batch_counter += 1
            self.loss_averages.append(loss)
        else:
            # save the data and pick one at random since we haven't hit the replay size
            self.data_set.add_sample(self.last_image, self.last_action, reward, False)
            int_action = self.randGenerator.randint(0, self.num_actions-1)

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
            int_action, considered = self.network.choose_action(phi, epsilon)
        else:
            int_action = self.randGenerator.randint(0, self.num_actions - 1)
            considered = False
        return int_action, considered


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

        self.step_counter += 1
        self.total_steps += 1
        total_time = time.time() - self.start_time

        if self.testing:
            self.episode_reward += reward
            if self.best_epoch_reward is None or self.episode_reward > self.best_epoch_reward:
                self.best_epoch_reward = self.episode_reward
                self.best_run_images = self.episode_images
                self.best_run_inner_images = self.episode_inner_images

                if self.best_score_ever is None or self.episode_reward > self.best_score_ever:
                    self.best_score_ever = self.episode_reward

            if not epoch_end or self.episode_counter == 0:
                # only collect stats for this episode if it didn't get truncated, or if it's the 
                # only one we are going to get for this testing epoch                
                self.episode_counter += 1
                self.total_reward += self.episode_reward
                self.record_run()
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
            if len(self.data_set) >= holdout_size:
                self.holdout_data = self.data_set.random_batch(holdout_size)[0]
            else:
                # one of those cases where we didn't train, we are just testing
                return 0

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
        self.current_epoch = epoch

    def _finish_epoch(self, epoch):
        self.agent_end(0, epoch_end=True)
        self._record_network(epoch)

    def _start_testing(self, epoch):
        self.testing = True
        self.current_epoch = epoch        
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
            epoch = int(in_message.split(" ")[1])
            self._start_testing(epoch)

        elif in_message.startswith("finish_testing"):
            epoch = int(in_message.split(" ")[1])            
            self._finish_testing(epoch)
        else:
            return "I don't know how to respond to your message"


    def _record_images(self, images, directory):
        if not os.path.exists(directory):
            os.mkdir(directory)

        for index, image in enumerate(images):
            full_name = os.path.join(directory, "frame%06d.png" % index)
            self.save_image(image, full_name)


    def record_best_run(self, epoch):
        if self.best_video:
            recording_directory = os.path.join(self.experiment_directory, "bestof%03d_%s" % (epoch, self.best_epoch_reward))
            self._record_images(self.best_run_images, recording_directory)

        if self.inner_video:
            recording_directory = os.path.join(self.experiment_directory, "bestinnerof%03d_%s" % (epoch, self.best_epoch_reward))
            self._record_images(self.best_run_inner_images, recording_directory)

    def record_run(self):
        if self.every_video:
            recording_directory = os.path.join(self.experiment_directory, "episode_%s_%s_%s" % (self.current_epoch, self.episode_counter, self.episode_reward))
            self._record_images(self.episode_images, recording_directory)

            if self.inner_video:
                recording_directory = os.path.join(self.experiment_directory, "innerepisode_%s_%s_%s" % (self.current_epoch, self.episode_counter, self.episode_reward))
                self._record_images(self.episode_inner_images, recording_directory)


    def _save_array(self, image, filename):
        # Need to swap the colour order since cv2 expects BGR
        if len(image.shape) == 3:
            # colour image
            cv2.imwrite(filename, image[:,:,::-1])
        else:
            cv2.imwrite(filename, image)

    def _show_phis(self, phi1, phi2):
        import matplotlib.pyplot as plt        
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+1)
            plt.imshow(phi1[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        for p in range(self.phi_length):
            plt.subplot(2, self.phi_length, p+5)
            plt.imshow(phi2[p, :, :], interpolation='none', cmap="gray")
            plt.grid(color='r', linestyle='-', linewidth=1)
        plt.show()



def addScriptArguments(parser=None, in_group=False):
    """
    Add arguments for this script to the passed parser, or create a parser if None is passed
    If in_group is true, add the parameters to a group
    """

    if parser is None:
        parser = argparse.ArgumentParser(description='Neural rl agent.')

    if in_group:
        group = parser.add_argument_group('Agent', 'Parameters passed to rl_glue_agent')
    else:
        group = parser

    group.add_argument("-v", "--verbose", dest="verbosity", default=0, action="count",
                      help="Verbosity.  Invoke many times for higher verbosity")

    parameters = group.add_mutually_exclusive_group(required=False)
    parameters.add_argument('--nips', dest="nips", action="store_true", default=False,
        help="""Set parameters like they used to work with this program when it was using
        DeepMind's NIPS paper's architecture (small network)""")
    parameters.add_argument('--nature', dest="nature", action="store_true", default=False,
        help="""Set parameters similar to DeepMind's Nature paper (large network) (this is the default)""")    

    group.add_argument("-g", '--game-name', dest="game_name", default=None,
        help='Name of the game')
    group.add_argument("-lr", '--learning-rate', dest="learning_rate", type=float, action=KnowWhereYouComeFrom,
        default=DefaultParameters.LearningRate,
        help='Learning rate (default: %(default)s)')
    group.add_argument("-d", '--discount', dest="discount_rate", type=float, default=DefaultParameters.DiscountRate,
        action=KnowWhereYouComeFrom,
        help='Discount rate (default: %(default)s)')
    group.add_argument("-m", '--momentum', dest="momentum", type=float, default=DefaultParameters.Momentum,
        action=KnowWhereYouComeFrom,
        help='Momentum term for Nesterov momentum (default: %(default)s)')    
    group.add_argument('--rms-decay', dest="rms_decay", type=float, default=DefaultParameters.RmsDecay, 
        action=KnowWhereYouComeFrom,
        help='Decay rate for rms_prop (default: %(default)s)')    
    group.add_argument('--replay-start-size', dest="replay_start_size", type=int, 
        default=DefaultParameters.ReplayStartSize,
        action=KnowWhereYouComeFrom,
        help='How many frames are needed till we start training (default: %(default)s)')    
    group.add_argument('--image-preparation', dest="image_preparation", choices=['crop', 'resize'],
        default=DefaultParameters.ImagePreparation,
        action=KnowWhereYouComeFrom,
        help='Whether to crop prior to resizing to keep the correct aspect ratio or to straight up resize the input image (default: %(default)s)')    
    group.add_argument('-b', '--batch-size', dest="batch_size", type=int, default=DefaultParameters.BatchSize,
        action=KnowWhereYouComeFrom,
        help='Batch size (default: %(default)s)')
    group.add_argument('--experiment-prefix', dest="experiment_prefix", type=str, default="",
        action=KnowWhereYouComeFrom,
        help='Experiment name prefix (default: %(default)s)')
    group.add_argument('--experiment-directory', dest="experiment_directory", type=str, default=None,
        action=KnowWhereYouComeFrom,
        help='Specify exact directory where to save output to (default: combination of prefix and game name and current date and parameters)')    
    group.add_argument("-n", '--nn-file', dest="nn_file", type=str, default=None,
        action=KnowWhereYouComeFrom,
        help='Pickle file containing trained net. (default: %(default)s)')
    group.add_argument("-p", '--pause', dest="pause", type=float, default=DefaultParameters.PauseTime,
        action=KnowWhereYouComeFrom,
        help='Amount of time to pause display while testing. (default: %(default)s)')
    group.add_argument("-es", '--epsilon-start', dest="epsilon_start", type=float,
        action=KnowWhereYouComeFrom,
        default=DefaultParameters.EpsilonStart,
        help='Starting value for epsilon. (default: %(default)s)')
    group.add_argument('--epsilon-min', dest="epsilon_min", type=float, default=DefaultParameters.EpsilonMin,
        action=KnowWhereYouComeFrom,
        help='Minimum epsilon. (default: %(default)s)')
    group.add_argument('--epsilon-decay', dest="epsilon_decay", type=float, default=DefaultParameters.EpsilonDecay,
        action=KnowWhereYouComeFrom,
        help='Number of steps to minimum epsilon. (default: %(default)s)')
    group.add_argument('-te', '--test-epsilon', dest="testing_epsilon", type=float, default=DefaultParameters.
        TestingEpsilon,
        action=KnowWhereYouComeFrom,
        help='Epsilon to use during testing (default: %(default)s)')        
    group.add_argument("-hl", '--history-length', dest="history_length", type=int, default=DefaultParameters.HistoryLength,
        action=KnowWhereYouComeFrom,
        help='History length (default: %(default)s)')
    group.add_argument('--max-history', dest="history_max", type=int, default=DefaultParameters.HistoryMax,
        action=KnowWhereYouComeFrom,
        help='Maximum number of steps stored (default: %(default)s)')
    group.add_argument('-tr', '--target-reset-frequency', dest="target_reset_frequency", type=int, default=DefaultParameters.TargetResetFrequency,
        action=KnowWhereYouComeFrom,
        help="How often to copy the current training network parameters into the estimator of goodness network, in frames. 0 to not use different networks (default: %(default)s)")
    group.add_argument('--no-video', dest="video", default=True, action="store_false",
        help='Do not make a "video" record of the best run in each testing epoch')    
    group.add_argument('--every-video', dest="every_video", default=False, action="store_true",
        help='Make a "video" record of every testing game played, not just the best in each testing epoch')        
    group.add_argument('--no-records', dest="recording", default=True, action="store_false",
        help='Do not record anything about the experiment (best games, epoch networks, test results, etc)')
    group.add_argument('--inner-video', dest="inner_video", default=False, action="store_true",
        help='Make a "video" of what the agent sees too')    


    return parser        



def main(args):
    """
    Mostly just read command line arguments here. We do this here
    instead of agent_init to make it possible to use --help from
    the command line without starting an experiment.
    """

    from logutils import setupLogging

    # Handle command line argument:
    parser = addScriptArguments()

    # ignore unknowns
    parameters, _ = parser.parse_known_args(args)

    setupLogging(parameters.verbosity)

    if not parameters.recording:
        best_video = epoch_network = learning_log = inner_video = every_video = False
    else:
        best_video = parameters.video
        inner_video = parameters.inner_video
        every_video = parameters.every_video
        epoch_network = learning_log = True

    if parameters.nips:
        default_parameters = NIPSParameters
    else:
        default_parameters = DefaultParameters
        


    AgentLoader.loadAgent(NeuralAgent(parameters.game_name,
        network_size=default_parameters.get_default(parameters, 'network_size'),
        learning_rate=default_parameters.get_default(parameters, 'learning_rate'),
        batch_size=default_parameters.get_default(parameters, 'batch_size'),
        discount_rate=default_parameters.get_default(parameters, 'discount_rate'),
        momentum=default_parameters.get_default(parameters, 'momentum'),
        rms_decay=default_parameters.get_default(parameters, 'rms_decay'),
        experiment_prefix=default_parameters.get_default(parameters, 'experiment_prefix'),
        experiment_directory=default_parameters.get_default(parameters, 'experiment_directory'),
        nn_file=default_parameters.get_default(parameters, 'nn_file'),
        pause=default_parameters.get_default(parameters, 'pause'),
        epsilon_start=default_parameters.get_default(parameters, 'epsilon_start'),
        epsilon_min=default_parameters.get_default(parameters, 'epsilon_min'),
        epsilon_decay=default_parameters.get_default(parameters, 'epsilon_decay'),
        testing_epsilon=default_parameters.get_default(parameters, 'testing_epsilon'),        
        history_length=default_parameters.get_default(parameters, 'history_length'),
        max_history=default_parameters.get_default(parameters, 'history_max'),
        replay_start_size=default_parameters.get_default(parameters, 'replay_start_size'),
        best_video=best_video,
        every_video=every_video,
        inner_video=inner_video,
        keep_epoch_network=epoch_network,
        learning_log=learning_log,
        target_reset_frequency=default_parameters.get_default(parameters, 'target_reset_frequency'),
        image_preparation=default_parameters.get_default(parameters, 'image_preparation')))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
