#!/bin/bash

# used to identify the job in squeue
#SBATCH --job-name=byu_swin3d_t

# specify node
#SBATCH --nodelist=argus[04]

# creates destination files for the output and error messages
# the format of the files are JOB_NAME.JOBID.out and JOB_NAME.JOBID.err
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --gres=gpu:1  # Request 1 GPU

#SBATCH --partition=long

export CUDA_HOME=/usr/local/cuda

echo "START on hostname"
hostname
export CUDA_VISIBLE_DEVICES=1
echo "CUDAVS",$CUDA_VISIBLE_DEVICES

pwd


source /opt/miniconda3/etc/profile.d/conda.sh
conda info --envs
conda activate motorenv

export PATH=/home/naamagav/.conda/envs/motorenv/bin:$PATH

python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.backends.cudnn.version())"
python -c "import torch; print(torch.__version__)"

conda info

python train.py -G=0 -C=r3d200 epochs=250 save_weights=True backbone="swin3d_t"

echo "END"
