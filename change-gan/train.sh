#!/usr/bin/env bash

now=$(date +"%Y%m%d_%H%M%S")
#JOB_NAME="autoconverter_$now"
JOB_NAME="change_gan_$now"

TRAINER_PACKAGE_PATH=change-gan
MAIN_TRAINER_MODULE=change-gan.main

BUCKET_NAME=mlcampjeju2017-mlengine
#JOB_DIR="gs://$BUCKET_NAME/autoconverter-7"
JOB_DIR="gs://$BUCKET_NAME/change-gan-bbox-1"
PACKAGE_STAGING_LOCATION="gs://$BUCKET_NAME/stage"
TRAIN_DIR="gs://$BUCKET_NAME/data-bbox"
EVAL_DIR="gs://$BUCKET_NAME/data-bbox"

LOCAL_JOB_DIR="/Users/SHYBookPro/Desktop/local-job-dir"
LOCAL_TRAIN_DIR="$LOCAL_JOB_DIR/data"
LOCAL_EVAL_DIR="$LOCAL_JOB_DIR/data"

REGION="asia-east1"
RUNTIME_VERSION="1.2"

if [ $1 = "cloud" ]; then
    gcloud ml-engine jobs submit training $JOB_NAME \
        --job-dir $JOB_DIR \
        --runtime-version $RUNTIME_VERSION \
        --module-name $MAIN_TRAINER_MODULE \
        --package-path $TRAINER_PACKAGE_PATH \
        --region $REGION \
        --config config.yaml \
        -- \
        --verbosity DEBUG  \
        --train-dir $TRAIN_DIR \
        --eval-dir $EVAL_DIR \
        --train-steps 200000 \
        --eval-steps 1 \
        --train-batch-size 1 \
        --eval-batch-size 3 \
        --learning-rate 0.0002
elif [ $1 = "local" ]; then
    gcloud ml-engine local train \
        --module-name $MAIN_TRAINER_MODULE \
        --package-path $MAIN_TRAINER_MODULE \
        -- \
        --job-dir $LOCAL_JOB_DIR \
        --verbosity DEBUG  \
        --train-dir $LOCAL_TRAIN_DIR\
        --eval-dir $LOCAL_EVAL_DIR \
        --train-steps 10 \
        --eval-steps 1 \
        --train-batch-size 1 \
        --eval-batch-size 3 \
        --learning-rate 0.0002
else
    echo "Usage: train.sh [cloud|local]"
    exit 1
fi
