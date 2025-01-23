#!/bin/bash
#SBATCH --job-name=STAMP_Preprocess
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

cd  # Path were STAMP is
path= # Path to the folder were wholslide images are saved
conf=config.yaml # Path to the config file (default: config.yaml)

wsiPathSTAMP=$(grep -E -w -i "wsi_dir*" $conf) # extrakt path
wsiPathSTAMP=$(echo $wsiPathSTAMP | cut -d':' -f2)
wsiPathSTAMP=$(echo $wsiPathSTAMP | cut -d' ' -f1) # cut the rest of
singularity run --bind $path:$wsiPathSTAMP STAMP_container.sif "stamp --config $conf preprocess"