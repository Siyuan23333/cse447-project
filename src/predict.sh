#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: bash src/predict.sh <path_to_test_data> <path_to_predictions>"
    exit 1
fi

TEST_DATA_PATH=$1
OUTPUT_PATH=$2

python /job/src/predict.py --test_data "$TEST_DATA_PATH" --test_output "$OUTPUT_PATH"