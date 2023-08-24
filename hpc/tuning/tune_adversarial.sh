#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name AND the job array --
#BSUB -J tune_adversarial[1-24]%8
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 5:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /zhome/ef/5/164617/master-thesis-results/outputs/logs/tune_adversarial_%J.out
#BSUB -e /zhome/ef/5/164617/master-thesis-results/outputs/logs/tune_adversarial_%J.err
# -- end of LSF options --

unset PYTHONHOME
unset PYTHONPATH
source $HOME/miniconda3/bin/activate

nvidia-smi
# Load the cuda module
module swap cuda/11.6

/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

seed=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2)
bm=(1 10 20 30 40 60 80 100 1 10 20 30 40 60 80 100 1 10 20 30 40 60 80 100)
s=${seed[$LSB_JOBINDEX - 1]}
bmi=${bm[$LSB_JOBINDEX - 1]}

python3 src/train_adversarial.py +conditional=false +e=0.2 +n=0.2 +bx=1 +bw=2 +bz=1 +by=1 +bhw=$bmi +bhz=$bmi +byz=1 +bc=1 ++seed=$s
python3 src/tune_bm.py -input outputs/train_adversarial/++seed=$s,+bc=1,+bhw=$bmi,+bhz=$bmi,+bw=2,+bx=1,+by=1,+byz=1,+bz=1,+conditional=false,+e=0.2,+n=0.2