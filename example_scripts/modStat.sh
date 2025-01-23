#!/bin/bash
#SBATCH --job-name=STAMP_Cross_Train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

cd  # Path were STAMP is
conf=config.yaml # Path to the config file (default: config.yaml)

singularity run STAMP_container.sif "stamp --config $conf crossval"
singularity run STAMP_container.sif "stamp --config $conf statistics"