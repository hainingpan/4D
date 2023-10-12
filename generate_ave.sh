#!/bin/bash
#SBATCH --mail-user=hnpanboa@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH -n 1
#SBATCH -c 10
#SBATCH -p kim
#SBATCH -t 10:00:00 
#SBATCH --mem=128G
python generate_ave.py