#!/bin/bash
set -e

echo "Building Inferencer for Termux ARM64 (QNN HTP)"

cd "$(dirname "$0")"

echo "go mod tidy..."
go mod tidy

echo "go build..."
go build -ldflags="-s -w" -o inferencer .

echo ""
echo "Done: ./inferencer"
echo ""
echo "Run (must be root):"
echo "  sudo LD_LIBRARY_PATH=\$HOME/.local/lib:/vendor/lib64 \\"
echo "    ADSP_LIBRARY_PATH=/vendor/lib/rfsa/adsp \\"
echo "    ./inferencer --host 192.168.x.x:8080"
