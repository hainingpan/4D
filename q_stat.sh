#!/bin/bash
#SBATCH --mail-user=hnpanboa@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH -n 12
#SBATCH -c 10
##SBATCH -N 5
#SBATCH -p kim
#SBATCH -t 10:00:00 
#SBATCH --mem=128G

# mpiexec -np 24 python -m mpi4py.futures q_Txy.py -o 1 -kind=all_peaks -drop=1 
# mpirun -np 40 python -m mpi4py.futures q_Txy.py -o thrs_3_test -kind=all_peaks -drop=1 -outlier

# mpiexec -np 24 python -m mpi4py.futures q_Txy.py -o 2 -kind=all_peaks -drop=0 -i /home/shared/STEM_recal_2/ 
# mpiexec -np 24 python -m mpi4py.futures q_Txy.py -o 2 -kind=all_peaks -drop=0 -outlier -i /home/shared/STEM_recal_2/


# mpiexec python -m mpi4py.futures q_Txy.py -o 1 -kind=remove_bragg -drop=1
# mpiexec python -m mpi4py.futures q_Txy.py -o 1 -kind=remove_bragg -drop=1 -outlier


# mpiexec python -m mpi4py.futures q_Txy.py -o 2 -kind=remove_bragg -drop=0 -i /home/shared/STEM_recal_2/
# mpiexec python -m mpi4py.futures q_Txy.py -o 2 -kind=remove_bragg -drop=0 -outlier -i /home/shared/STEM_recal_2/

# mpirun -np 40 python -m mpi4py.futures q_Txy.py -o 1 -kind=count_pts -drop=1 
# mpirun -np 40 python -m mpi4py.futures q_Txy.py -o 1 -kind=count_pts -drop=1 -outlier

# mpirun -np 40 python -m mpi4py.futures q_Txy.py -o 2 -kind=count_pts -drop=1 
# mpirun -np 40 python -m mpi4py.futures q_Txy.py -o 2 -kind=count_pts -drop=1 -outlier

# mpirun -np 40 python -m mpi4py.futures q_Txy.py -i /home/shared/STEM_sample_2/ -o sample_2 -kind=count_pts -drop=0 
# srun python -m mpi4py.futures q_Txy.py -i /share/kim/STEM_sample_2/ -o sample_2_auto_bragg -kind=count_pts -drop=0 -outlier -auto
srun python -m mpi4py.futures q_Txy.py -i /share/kim/STEM_sample_2/ -o sample_2_auto_bragg_iter -kind=count_pts -drop=0 -outlier -auto

# srun python -m mpi4py.futures q_Txy.py -i /share/kim/STEM_sample_3/ -o sample_3_auto_bragg -kind=count_pts -drop=0 -outlier -auto
# srun python -m mpi4py.futures q_Txy.py -i /share/kim/STEM_sample_3/ -o sample_3_auto_bragg_relocate -kind=count_pts -drop=0 -outlier -auto
# srun python -m mpi4py.futures q_Txy.py -i /share/kim/STEM_sample_3/ -o sample_3_auto_bragg_iter -kind=count_pts -drop=0 -outlier -auto