#!/usr/bin/env bash
# Detector models for subject priority (faces + animals). Verifies SHA-256 after download.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models

YUNET_URL="https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
YUNET_SHA="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
curl -fsSL -o models/face_detection_yunet_2023mar.onnx "$YUNET_URL"
echo "$YUNET_SHA  models/face_detection_yunet_2023mar.onnx" | sha256sum -c -

YOLOX_URL="https://huggingface.co/opencv/object_detection_yolox/resolve/main/object_detection_yolox_2022nov.onnx"
YOLOX_SHA="c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
curl -fsSL -o models/object_detection_yolox_2022nov.onnx "$YOLOX_URL"
echo "$YOLOX_SHA  models/object_detection_yolox_2022nov.onnx" | sha256sum -c -

CAT_CASCADE_URL="https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalcatface.xml"
CAT_CASCADE_SHA="ac2bac934ef24284ef8a2b2e9d8e57eef84ac1d6943b4d11e5c9e8584dc069c8"
curl -fsSL -o models/haarcascade_frontalcatface.xml "$CAT_CASCADE_URL"
echo "$CAT_CASCADE_SHA  models/haarcascade_frontalcatface.xml" | sha256sum -c -