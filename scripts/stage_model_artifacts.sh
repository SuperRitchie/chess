#!/usr/bin/env bash
set -euo pipefail

buffer_path="$1"
start_sha="${GITHUB_SHA:-$(git rev-parse HEAD)}"
artifact_dir="$(mktemp -d)"
trap 'rm -rf "$artifact_dir"' EXIT

mkdir -p "$artifact_dir/public" "$artifact_dir/ml/checkpoints" "$(dirname "$artifact_dir/$buffer_path")" "$artifact_dir/ml/data"
cp -a public/nn "$artifact_dir/public/"
cp -a ml/checkpoints/. "$artifact_dir/ml/checkpoints/"
cp -a "$buffer_path" "$artifact_dir/$buffer_path"
cp -a ml/data/fixed_eval_set_v3.json "$artifact_dir/ml/data/fixed_eval_set_v3.json"
cp -a ml/training_history.json "$artifact_dir/ml/training_history.json"

git fetch origin master

# Do not publish artifacts trained with stale code on top of a newer training implementation.
if ! git diff --quiet "$start_sha..origin/master" -- \
  'ml/*.py' 'ml/tests/**' 'src/**' requirements.txt package.json package-lock.json scripts .github/workflows \
  'public/nn/**' 'ml/checkpoints/**'; then
  echo "::error::Training or inference code changed while this job was running. Rerun from the latest master."
  exit 2
fi

git reset --hard origin/master
git clean -fd

candidate_accepted="$(python -c 'import json,sys; data=json.load(open(sys.argv[1],encoding="utf-8")); print(str(bool(data and data[-1].get("candidate_accepted"))).lower())' "$artifact_dir/ml/training_history.json")"

mkdir -p public ml/checkpoints "$(dirname "$buffer_path")" ml/data
if [[ "$candidate_accepted" == "true" ]]; then
  rm -rf public/nn ml/checkpoints
  cp -a "$artifact_dir/public/nn" public/
  mkdir -p ml/checkpoints
  cp -a "$artifact_dir/ml/checkpoints/." ml/checkpoints/
fi
cp -a "$artifact_dir/$buffer_path" "$buffer_path"
cp -a "$artifact_dir/ml/data/fixed_eval_set_v3.json" ml/data/fixed_eval_set_v3.json
python scripts/merge_training_history.py ml/training_history.json "$artifact_dir/ml/training_history.json"
