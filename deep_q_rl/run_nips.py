# ! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning with parameters that
are consistent with:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013



/!\ HERE ARE ALL THE EXPLANATION OF THE MODIFICATION I DID TO TEST /!\

Most of the modifications are in ale_experiment.

Basically, what I did was replace the concept of epoch by the concept of 
episode. Because renaming is a bit painful, I just left epoch but if you read 
the code, please remember that you should always replace it by episode in the 
nips and nature paper sense !



"""

import launcher
import sys

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 50000
    EPOCHS = 100
    STEPS_PER_TEST = 10000

    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = '../../roms/'
    ROM = 'breakout_super.bin'
    MODE = 1
    FRAME_SKIP = 4
    REPEAT_ACTION_PROBABILITY = 0

    # ----------------------
    # Agent/Network Parameters
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    BATCH_ACCUMULATOR = 'mean'
    LEARNING_RATE = .0002
    DISCOUNT = .95
    RMS_DECAY = .99 # (Rho)
    RMS_EPSILON = 1e-6
    MOMENTUM = 0
    CLIP_DELTA = 0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = 'nips_dnn'
    FREEZE_INTERVAL = -1
    REPLAY_START_SIZE = 100
    RESIZE_METHOD = 'crop'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE = 'true'
    MAX_START_NULLOPS = 0
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False
    USE_DOUBLE = False


    # EPOCHS = 1
    # LEARNING_RATE = 0
    # EPSILON_START = 0
    # EPSILON_MIN = 0
    # EPSILON_DECAY = 0


if __name__ == '__main__':
    launcher.launch(sys.argv[1:], Defaults, __doc__)
