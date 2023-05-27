#!/usr/bin/env bash

#CONFIG=$1

#python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch

export NCCL_P2P_DISABLE=1

python setup.py develop --no_cuda_ext

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4398 basicsr/train.py -opt Deraining/Options/Deraining_Restormer.yml --launcher pytorch

