#!/bin/bash
#SBATCH -J MS_SEGUE
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 05:00:00 # Runtime
#SBATCH --mem 5000 # Memory
#SBATCH -p conroy_priority,shared,itc_cluster,serial_requeue # Partition to submit to
#SBATCH --constraint='intel'
#SBATCH -o /n/holyscratch01/conroy_lab/vchandra/segue/logs/test/VX/%a.out
#SBATCH -e /n/holyscratch01/conroy_lab/vchandra/segue/logs/test/VX/%a.err
#SBATCH --array=0-1

source /n/home03/vchandra/warmup.sh

cd /n/home03/vchandra/outerhalo/09_sdss5/pipeline/99_fit_segue/
echo 'CPU USED: ' 
cat /proc/cpuinfo | grep 'model name' | head -n 1
echo 'QUEUE NAME:' 
echo $SLURM_JOB_PARTITION
echo 'NODE NAME:' 
echo $SLURMD_NODENAME 

python 04_runms_star.py --catalog=test --ind=$SLURM_ARRAY_TASK_ID --version=VX --npoints=750 --skipfit=0