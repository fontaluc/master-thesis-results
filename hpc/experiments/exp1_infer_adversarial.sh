#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J exp1_infer_adversarial[1-18]%6
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o /zhome/ef/5/164617/master-thesis-results/outputs/logs/exp1_infer_adversarial_%J.out
#BSUB -e /zhome/ef/5/164617/master-thesis-results/outputs/logs/exp1_infer_adversarial_%J.err

unset PYTHONHOME
unset PYTHONPATH
source $HOME/miniconda3/bin/activate

seed=(0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2 2)
cond=(true true true false false false true true true false false false true true true false false false)
e=(0.05 0.2 0.5 0.05 0.2 0.5 0.05 0.2 0.5 0.05 0.2 0.5 0.05 0.2 0.5 0.05 0.2 0.5)
python3 src/inference.py -input outputs/train_adversarial/++seed=${seed[$LSB_JOBINDEX - 1]},+bc=1,+bhw=10,+bhz=10,+bw=10,+bx=1,+by=10,+byz=1,+bz=1,+conditional=${cond[$LSB_JOBINDEX - 1]},+e=${e[$LSB_JOBINDEX - 1]} -visualize False