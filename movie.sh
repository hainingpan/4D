#!/bin/bash
#SBATCH --mail-user=hnpanboa@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH -c 20
#SBATCH -p kim
#SBATCH -t 10:00:00 
#SBATCH --mem=128G

srun python movie.py --filename ave_cluster_real_space_auto_no_vac_3 --model_filename count_pts_outlier_auto_3.pickle --movie --figure --array --filetype_list mp4 gif --dpi 100

