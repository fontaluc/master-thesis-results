#!/bin/sh 
for s in 0 1 2
do 
    python3 src/inference.py -input outputs/train_baseline/++seed=$s,+bw=2,+bx=1,+by=1,+bz=1,+e=0.2,+n=0.2
    for c in true false
    do
        python3 src/inference.py -input outputs/train_mmd/++seed=$s,+bmw=20,+bmz=20,+bw=2,+bx=1,+by=1,+bz=1,+conditional=$c,+e=0.2,+n=0.2
        python3 src/inference.py -input outputs/train_adversarial/++seed=$s,+bc=1,+bhw=20,+bhz=20,+bw=2,+bx=1,+by=1,+byz=1,+bz=1,+conditional=$c,+e=0.2,+n=0.2
    done
done