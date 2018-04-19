#!/bin/sh
pyenv shell anaconda2-5.1.0
PYTHONPATH=~/.pyenv/versions/anaconda2-5.1.0/lib/python2.7/site-packages:/usr/lib/python2.7/dist-packages:$PYTHONPATH roslaunch object_detector run.launch

