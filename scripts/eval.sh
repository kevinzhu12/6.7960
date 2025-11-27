#!/bin/bash
#SBATCH --account=vision-torralba-urops-meng
#SBATCH --partition=vision-shared-v100
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH -J cod10k-v3-enhanced-eval                 # Job name
#SBATCH -o /data/vision/torralba/selfmanaged/torralba/projects/gdaras/collaborators/kbzhu/diffusers-exploration/watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --mem=500G                   # server memory requested (per node)
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

cd /data/vision/torralba/selfmanaged/torralba/projects/gdaras/collaborators/kbzhu/diffusers-exploration

# activate conda environment
source /data/vision/torralba/selfmanaged/torralba/projects/gdaras/collaborators/kbzhu/miniforge3/etc/profile.d/conda.sh
conda activate diffusers-exploration

# enhance images
python enhance_images.py

# run evaluation
python clip_eval.py

# deactivate conda environment
conda deactivate

echo "Evaluation complete"