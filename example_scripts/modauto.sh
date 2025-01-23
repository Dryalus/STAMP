#!/bin/bash
#SBATCH --job-name=STAMP_Cross_Train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

# Automated Training with changeing Hyperparameters

cd  # Path were STAMP is
path= # Path were he saves the data
conf=config.yaml # Path to the config file (default: config.yaml)

for lm in {015..025} # Change the max learnrate Hyperparameter
do
	mkdir $path/out$lm
	mkdir $path/model_statistics/ms$lm
	sed -i 's/lr_max:\s0.[0-9]*/lr_max: 0.'${lm}'/g' $conf
	
	for dR in {002..006} # Change the decay rate of the wights in the training model
	do
		sed -i 's/wd:\s0.[0-9]*/wd: 0.'${dR}'/g' $conf
		echo wight decy rate $dR
		singularity run STAMP_container.sif "stamp --config $conf crossval"
		singularity run STAMP_container.sif "stamp --config $conf statistics"
		mv $path/output $path/out$lm/output$dR
		mv $path/model_statistics/model_statistics $path/model_statistics/ms$lm/model_statistics$dR
		mkdir $path/output
	done
done