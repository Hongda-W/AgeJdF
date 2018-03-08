#!/bin/bash

#SBATCH -J Fit_JdF
#SBATCH -o Fit_JdF_%j.out
#SBATCH -e Fit_JdF_%j.err
#SBATCH -N 1
#SBATCH --partition=shas
#SBATCH --qos=long
#SBATCH --ntasks-per-node=24
#SBATCH --time=5:00:00
#SBATCH --mem=MaxMemPerNode

cd /projects/howa1663/Code/AgeJdF
python run_fit_age.py
