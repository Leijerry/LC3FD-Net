#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -o ppfPointnet.log
python main.py --mode train --batch_size 24 --resume=Flase
