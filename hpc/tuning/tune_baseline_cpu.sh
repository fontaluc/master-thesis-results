#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J tune_baseline_cpu[1-3]
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
#BSUB -o /zhome/ef/5/164617/master-thesis-results/outputs/logs/tune_baseline_cpu_%J.out
#BSUB -e /zhome/ef/5/164617/master-thesis-results/outputs/logs/tune_baseline_cpu_%J.err

unset PYTHONHOME
unset PYTHONPATH
source $HOME/miniconda3/bin/activate

# by=(0 0 0)
# byi=${by[$LSB_JOBINDEX - 1]}
seed=(0 1 2)
# s=${seed[$LSB_JOBINDEX - 1]}

python src/tune_bm.py -input outputs/train_baseline/++seed=${seed[$LSB_JOBINDEX - 1]},+bw=2,+bx=1,+by=1,+bz=1,+e=0.2,+n=0.2