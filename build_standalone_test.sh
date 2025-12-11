#!/bin/bash
# Build standalone network probe test
# This only rebuilds network_probe.cc and dependencies

set -e

echo "🔨 Building standalone network probe test..."
echo "This is much faster than full XLA build!"
echo ""

bazel build --config=rocm \
  --compilation_mode=dbg \
  --copt=-O1 \
  --copt=-g \
  //xla/backends/profiler/gpu:network_probe_standalone_test

if [ $? -eq 0 ]; then
  echo ""
  echo "✅ Build successful!"
  echo ""
  echo "Binary location:"
  BINARY=$(bazel cquery --config=rocm --output=files //xla/backends/profiler/gpu:network_probe_standalone_test 2>/dev/null)
  echo "  $BINARY"
  echo ""
  echo "To run 2-node test:"
  echo "  Terminal 1: $BINARY --node_id=0 --num_nodes=2"
  echo "  Terminal 2: $BINARY --node_id=1 --num_nodes=2"
  echo ""
  echo "Or use: ./run_2node_test.sh"
else
  echo "❌ Build failed"
  exit 1
fi


