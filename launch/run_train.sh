#!/usr/bin/env bash

python2 train.py \
--data_dir /home/versa/fuchen/ganimation/sample_dataset \
--name st \
--batch_size 25 \
--train_ids_file "train_ids_0929.csv" \
--test_ids_file "test_ids_0929.csv" \
--images_folder "imgs_all" \
--aus_file "aus_1009.pkl" \
--load_epoch 56 \
--nepochs_no_decay 60 \
--nepochs_decay 40 \
--n_threads_train 0
