#!/bin/bash
#SBATCH --job-name="GPUTest"
#SBATCH -p GPU
#SBATCH --gres=gpu:v100-32:8
#SBATCH --reservation=GPUcis230067p
#SBATCH -t 08:00:00


#echo everything to stdout
set -x

#show gpu info
nvidia-smi
source hackauton/bin/activate
#load some software and run a script within your job
#https://www.psc.edu/resources/software/anaconda/

python3 finetune.py --bs 32 --net CRATE_base --classes 5 --opt adamW  --lr 5e-5 --n_epochs 200 --randomaug 1 --data medical --ckpt_dir /jet/home/guntakan/crate-emergence-notebooks/crate-demo.pth
#python3 main.py --batch-size 256 --arch CRATE_base  --optimizer Lion  --lr 5e-5 --epochs 200 --weight-decay 0.05 --data /ocean/projects/cis230067p/cteh/cxr14