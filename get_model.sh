#!/usr/bin/env bash
# Downloads the YuNet face detector (OpenCV Zoo, MIT license) into ./models and checks its hash.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
URL="https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
SHA="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
curl -fsSL -o models/face_detection_yunet_2023mar.onnx "$URL"
echo "$SHA  models/face_detection_yunet_2023mar.onnx" | sha256sum -c -
