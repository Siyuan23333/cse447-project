#!/usr/bin/env bash
set -e
set -v
# python src/myprogram.py test --work_dir work --test_data $1 --test_output $2

TEST_DATA=$1
OUTPUT_FILE=$2
WORK_DIR="work"

python myprogram.py test --work_dir "$WORK_DIR" --test_data "$TEST_DATA" --test_output "$OUTPUT_FILE"

