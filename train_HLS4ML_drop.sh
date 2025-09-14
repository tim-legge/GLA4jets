#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_HLS4ML`
DATADIR=${DATADIR_HLS4ML}
[[ -z $DATADIR ]] && DATADIR='./datasets/HLS4ML'
# set a comment via `COMMENT`
suffix=${COMMENT}

# PN, PFN, PCNN, ParT
model=$1
extraopts=""
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/ParT_drop.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-3"
elif [[ "$model" == "ParT-FineTune" ]]; then
    modelopts="networks/example_ParticleTransformer_finetune.py --use-amp --optimizer-option weight_decay 0.01"
    lr="1e-4"
    extraopts="--optimizer-option lr_mult (\"fc.*\",50) --lr-scheduler none --load-model-weights models/ParT_kin.pt"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    lr="1e-2"
elif [[ "$model" == "PN-FineTune" ]]; then
    modelopts="networks/example_ParticleNet_finetune.py"
    lr="1e-3"
    extraopts="--optimizer-option lr_mult (\"fc_out.*\",50) --lr-scheduler none --load-model-weights models/ParticleNet_kin.pt"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    lr="2e-2"
    extraopts="--batch-size 4096"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    lr="2e-2"
    extraopts="--batch-size 4096"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin, kinplus"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="kin"
if ! [[ "${FEATURE_TYPE}" =~ ^(kin|kinplus)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

weaver \
    --data-train "${DATADIR}/train/train-parquet/jetImage_0*.parquet" \
    "${DATADIR}/train/train-parquet/jetImage_1*.parquet" \
    "${DATADIR}/train/train-parquet/jetImage_2*.parquet" \
    "${DATADIR}/train/train-parquet/jetImage_3*.parquet" \
    "${DATADIR}/train/train-parquet/jetImage_4*.parquet" \
    "${DATADIR}/train/train-parquet/jetImage_5*.parquet" \
    --data-val "${DATADIR}/val/val-parquet/*.parquet" \
    --data-test "${DATADIR}/train/train-parquet/jetImage_6*.parquet" \
    --data-config data/HLS4ML/hls4ml_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/HLS4ML/${model}/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 1 --in-memory \
    --batch-size 512 --samples-per-epoch $((2400 * 512)) --samples-per-epoch-val $((800 * 512)) --num-epochs 20 --gpus 0 \
    --start-lr $lr --optimizer ranger --log logs/HLS4ML_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard HLS4ML_${FEATURE_TYPE}_${model}${suffix} \
    ${extraopts} "${@:3}"
