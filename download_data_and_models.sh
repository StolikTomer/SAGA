#!/bin/bash

# define a download function
function google_drive_download()
{
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

function open_directory()
{
  if [ -d $1 ]; then
    echo "$1 already exists"
  else
    echo "creating $1 directory"
    mkdir $1
  fi
}

for name in "coma" "smal"
do
  if [ $name = "coma" ]; then
    dataset_id="16SwLIZj5ZCi0HpGMsU9Z7oTknGyneyqV"
    spectral_id="1otxMDf6NpK_-EnQVxpGp5ORSkr1om8rg"
    models_id="13wU79FnE34SaMf6_acX0ER_PD2rNV9nL"
    transfer_mlp_ae_id="1oLd4OTmCxsbrX4BJMJem6W6gnHC871f7"
  else
    dataset_id="1xwXhEh7EAGTEPU2hVtUt9Fb94p8tFT6a"
    spectral_id="1gKLTgzcYM07p1akwxyG_UDljfEz8QJC8"
    models_id="1Ax_1p2pG2u5swiSet8D4NDw5aBjR5EWN"
  fi

  # opening directories
  for path in $name "$name/data" "${name}/models" "${name}/models/autoencoders" "${name}/models/classifiers"
  do
    open_directory $path
  done

  # download dataset
  dir_data_raw="$name/data/raw"
  open_directory $dir_data_raw
  google_drive_download $dataset_id raw_dataset.zip
  mv raw_dataset.zip $dir_data_raw
  cd $dir_data_raw
  unzip raw_dataset.zip
  rm raw_dataset.zip
  cd ../../../

  # download spectral
  dir_data_spectral="$name/data/spectral"
  open_directory $dir_data_spectral
  google_drive_download $spectral_id spectral.zip
  mv spectral.zip $dir_data_spectral
  cd $dir_data_spectral
  unzip spectral.zip
  rm spectral.zip
  cd ../../../

  # download models
  dir_ae="$name/models/autoencoders/official"
  dir_cls="$name/models/classifiers/official"
  open_directory $dir_ae
  open_directory $dir_cls
  google_drive_download $models_id models.zip
  unzip models.zip
  mv AE_1999.h5 $dir_ae
  mv CLS_999.h5 $dir_cls
  rm models.zip

  if [ $name = "coma" ]; then
    dir_transfer_ae="$name/models/autoencoders/transfer_mlp"
    open_directory $dir_transfer_ae
    google_drive_download $transfer_mlp_ae_id transfer_ae.zip
    unzip transfer_ae.zip
    mv AE_1999.h5 $dir_transfer_ae
    rm transfer_ae.zip
  fi
done

