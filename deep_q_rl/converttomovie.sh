#!/bin/bash
if [ $# -lt 1 ]
then
   output=output.mp4
else
   output=$1
   if [ $# -lt 2 ]
   then
      fps=15
   else
      fps=$2
   fi
fi

avconv -r $fps -i frame%06d.png $output

