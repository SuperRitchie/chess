#!/usr/bin/env bash
set -euo pipefail

buffer_path="$1"
artifact_dir="$(mktemp -d)"
mkdir -p "$artifact_dir/public" "$artifact_dir/ml/data" "$artifact_dir/ml/checkpoints"

cp -a public/nn "$artifact_dir/public/"
cp -a ml/checkpoints/. "$artifact_dir/ml/checkpoints/"
cp -a "$buffer_path" "$artifact_dir/$buffer_path"
cp -a ml/data/fixed_eval_set.json "$artifact_dir/ml/data/fixed_eval_set.json"
cp -a ml/training_history.json "$artifact_dir/ml/training_history.json"

git fetch origin master
git reset --hard origin/master
git clean -fd

mkdir -p public ml/checkpoints ml/data
cp -a "$artifact_dir/public/nn" public/
cp -a "$artifact_dir/ml/checkpoints/." ml/checkpoints/
cp -a "$artifact_dir/$buffer_path" "$buffer_path"
cp -a "$artifact_dir/ml/data/fixed_eval_set.json" ml/data/fixed_eval_set.json
cp -a "$artifact_dir/ml/training_history.json" ml/training_history.json
