#!/bin/bash
if [ $# -lt 1 ]
then
   output=output.mp4
else
   output=$1
fi
avconv -r 15 -i frame%06d.png $output

