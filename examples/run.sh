#!/bin/bash

screen -dmS a bash -c "export CUDA_VISIBLE_DEVICES=0;python -i bird.py -L 0";
screen -dmS b bash -c "export CUDA_VISIBLE_DEVICES=1;python -i bird.py -L 4";
screen -dmS c bash -c "export CUDA_VISIBLE_DEVICES=2;python -i bird.py -L 8";




