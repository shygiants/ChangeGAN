#!/usr/bin/env bash

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="autoconverter_$now"

TRAINER_PACKAGE_PATH=change-gan
MAIN_TRAINER_MODULE=change-gan.main

BUCKET_NAME=mlcampjeju2017-mlengine
JOB_DIR="gs://$BUCKET_NAME/autoconverter-4"
PACKAGE_STAGING_LOCATION="gs://$BUCKET_NAME/stage"
TRAIN_DIR="gs://$BUCKET_NAME/data"
EVAL_DIR="gs://$BUCKET_NAME/eval"

REGION="asia-east1"
RUNTIME_VERSION="1.2"

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
    --train-steps 100000 \
    --eval-steps 1 \
    --train-batch-size 200 \
    --eval-batch-size 3