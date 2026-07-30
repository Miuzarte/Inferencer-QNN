#!/bin/bash
set -e

echo "Building Inferencer for Linux ARM64 (QNN HTP)"

export CGO_ENABLED=1
export GOOS=linux
export GOARCH=arm64

LD_LIBRARY="/root/.local/lib/onnxruntime/lib:/root/.local/lib/onnxruntime-qnn"

cd "$(dirname "$0")"

echo "go mod tidy..."
go mod tidy

echo "go build..."
go build -ldflags="-s -w" -o inferencer .

echo "Done: ./inferencer"
echo "Run: LD_LIBRARY_PATH=$LD_LIBRARY ADSP_LIBRARY_PATH=/vendor/lib/rfsa/adsp ./inferencer --host HOST:PORT"
