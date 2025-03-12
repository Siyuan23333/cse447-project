#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Siyuan Ge,jamesgsy\nAudrey Sein Aye,aseinaye" > submit/team.txt

# make predictions on example data submit it in pred.txt
bash src/predict.sh example/input.txt submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r submit.zip submit
