"""
Database of which part of the screen we capture during play. This only determines vertical area, since 
we capture the whole horizontal space

"""

from collections import defaultdict

# Since we are going to capture a square section of the screen, we need to specify which part 
# of the vertical sector to select. The list defaults to 'bottom', which means just grab the 
# bottom section. Other posibilities are 'top', 'centre', or a number. The number specifies
# the number of pixels from the top of the screen, if positive, and from the bottom if negative
# Pixels are measured from the original 160 x 210 screen

PlayingArea = defaultdict(lambda: 'bottom',
    {
    'breakout' : -14,
    'pong' : -16,
    'space_invaders' : -14,
    'seaquest' : -20,

    })