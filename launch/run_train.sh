#!/usr/bin/env bash

python2 train.py \
--data_dir /versa/fuchen/ganimation/sample_dataset \
--name st \
--batch_size 81 \
--train_ids_file "train_ids_0929.csv" \
--test_ids_file "test_ids_0929.csv" \
--images_folder "imgs_all" \
--aus_file "aus_1009.pkl" \
--load_epoch 1 \
--nepochs_no_decay 60 \
--nepochs_decay 40 \
--n_threads_train 1 \
--train_G_every_n_iterations 1 \
--lr_G 0.0001 \
--G_adam_b1 0 \
--G_adam_b2 0.9 \
--lr_D 0.0004 \
--D_adam_b1 0 \
--D_adam_b2 0.9
