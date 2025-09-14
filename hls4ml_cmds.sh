#!/bin/bash

git clone https://github.com/tim-legge/ParT_Interpretability && \
pip install 'weaver-core>=0.4' && \
mkdir hls4ml_training && \
cd ParT_Interpretability/ && \
mkdir datasets datasets/hls4ml && \
./get_datasets.py HLS4ML -d datasets/hls4ml && \
python batch_convert.py datasets/hls4ml/train/train datasets/hls4ml/HLS4ML/train/train-parquet && \ 
python batch_convert.py datasets/hls4ml/val/val datasets/hls4ml/HLS4ML/val/val-parquet && \
./train_HLS4ML.sh ParT kin --model-prefix /home/jovyan/hls4ml_training --num-epochs 4 --batch-size 32 --fetch-step 0.15