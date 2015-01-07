"""
Database of actions available to each game.
Since humans would be limited by the hardware in what actions are available (ie can't up-left on a paddle), I think it'd be 'fair' to limit
the program similarly
"""

# This is a mapping from game name to ALE's identity of available actions
# If the game uses all 18 actions, there's no need to put an entry here

# ALE's actions are (from src/common/Constants.h):

from collections import defaultdict

PLAYER_A_NOOP           = 0
PLAYER_A_FIRE           = 1
PLAYER_A_UP             = 2
PLAYER_A_RIGHT          = 3
PLAYER_A_LEFT           = 4
PLAYER_A_DOWN           = 5
PLAYER_A_UPRIGHT        = 6
PLAYER_A_UPLEFT         = 7
PLAYER_A_DOWNRIGHT      = 8
PLAYER_A_DOWNLEFT       = 9
PLAYER_A_UPFIRE         = 10
PLAYER_A_RIGHTFIRE      = 11
PLAYER_A_LEFTFIRE       = 12
PLAYER_A_DOWNFIRE       = 13
PLAYER_A_UPRIGHTFIRE    = 14
PLAYER_A_UPLEFTFIRE     = 15
PLAYER_A_DOWNRIGHTFIRE  = 16
PLAYER_A_DOWNLEFTFIRE   = 17


GameActions = defaultdict(lambda: [PLAYER_A_NOOP,PLAYER_A_FIRE,PLAYER_A_UP,PLAYER_A_RIGHT,PLAYER_A_LEFT,PLAYER_A_DOWN,PLAYER_A_UPRIGHT,PLAYER_A_UPLEFT,PLAYER_A_DOWNRIGHT,PLAYER_A_DOWNLEFT,PLAYER_A_UPFIRE,PLAYER_A_RIGHTFIRE,PLAYER_A_LEFTFIRE,PLAYER_A_DOWNFIRE,PLAYER_A_UPRIGHTFIRE,PLAYER_A_UPLEFTFIRE,PLAYER_A_DOWNRIGHTFIRE,PLAYER_A_DOWNLEFTFIRE], {
    'breakout' : [PLAYER_A_NOOP, PLAYER_A_FIRE, PLAYER_A_RIGHT, PLAYER_A_LEFT, PLAYER_A_RIGHTFIRE, PLAYER_A_LEFTFIRE]
    }
)

