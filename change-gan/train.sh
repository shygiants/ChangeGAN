#!/usr/bin/env bash

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="autoconverter_$now"

TRAINER_PACKAGE_PATH=change-gan
MAIN_TRAINER_MODULE=change-gan.main

BUCKET_NAME=mlcampjeju2017-mlengine
JOB_DIR="gs://$BUCKET_NAME/autoconverter-2"
PACKAGE_STAGING_LOCATION="gs://$BUCKET_NAME/stage"
DATASET_DIR="gs://$BUCKET_NAME/data"

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
    --dataset-dir $DATASET_DIR \
    --eval-steps 1