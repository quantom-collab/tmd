#!/bin/bash

# SIDIS Computation Build and Run Script

set -e

echo "Building SIDIS Computation Tools..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
SOURCE_DIR="$SCRIPT_DIR"

# Create and enter build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
cmake "$SOURCE_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$SOURCE_DIR/install"

# Build
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "✅ Build completed successfully!"
echo ""
echo "📁 Executables located in: $BUILD_DIR"
echo "   - SIDISCrossSection (original interface)"
echo "   - SIDISCrossSection_new (custom kinematics interface)"

echo ""
echo "🚀 Example usage for new executable:"
echo "   cd $BUILD_DIR"
echo "   ./SIDISCrossSection_new ../../inputs/config.yaml ../../inputs/kinematics.yaml output"
echo ""
echo "📖 For help:"
echo "   ./SIDISCrossSection_new --help"
echo ""
echo "🔧 To install system-wide:"
echo "   make install"
