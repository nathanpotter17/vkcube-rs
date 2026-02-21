#!/bin/bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global build mode (set only for engine builds)
BUILD_MODE=""

# Shader list
shaders=(
    "basic.vert:basic.vert.spv"
    "basic.frag:basic.frag.spv"
)

set_avx2() {
    # Check if AVX2 is already set in RUSTFLAGS
    if [[ "$RUSTFLAGS" == *"target-feature=+avx2"* ]]; then
        echo -e "${GREEN}AVX2 already enabled${NC}"
        return
    fi
    
    echo -e "${YELLOW}⚠ WARNING: This build requires AVX2 CPU instructions${NC}"
    echo -e "Most modern CPUs (2013+) support this, but older systems may not."
    read -p "Continue with AVX2 enabled? [Y/n]: " confirm
    case "$confirm" in
        [Nn]) echo -e "${RED}Build cancelled.${NC}"; exit 0 ;;
        *) export RUSTFLAGS="-C target-feature=+avx2"
           echo -e "${GREEN}AVX2 enabled${NC}" ;;
    esac
}
 
compile_shaders() {
    echo -e "${CYAN}Compiling shaders...${NC}"
    mkdir -p shaders/compiled
    for shader in "${shaders[@]}"; do
        IFS=':' read -r src dst <<< "$shader"
        echo -e "Compiling $src..."
        if ! glslc "shaders/$src" -o "shaders/compiled/$dst" --target-env=vulkan1.1 --target-spv=spv1.3; then
            echo -e "${RED}✗ Shader compilation failed for $src${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓${NC} $src"
    done
}

build_engine() {
    set_avx2
    compile_shaders
    
    echo -e "${CYAN}Building engine ($BUILD_MODE)...${NC}"
    
    if [ "$BUILD_MODE" = "release" ]; then
        cargo build --release
    else
        cargo build
    fi
    
    echo -e "${GREEN}✓ Engine build complete${NC}"
}

build_with_warnings_suppressed() {
    set_avx2
    # compile_shaders
    echo -e "${CYAN}Building with warnings suppressed (release)...${NC}"
    RUSTFLAGS="-C target-feature=+avx2 -A warnings" cargo build --release
    echo -e "${GREEN}✓ Build complete${NC}"
}

run_engine() {
    local binary="./target/${BUILD_MODE:-release}/simmer"
    
    if [ ! -f "$binary" ]; then
        if [ -f "./target/release/simmer" ]; then
            binary="./target/release/simmer"
            echo -e "${GREEN}Found release build${NC}"
        elif [ -f "./target/debug/simmer" ]; then
            binary="./target/debug/simmer"
            echo -e "${YELLOW}Found debug build${NC}"
        else
            echo -e "${RED}Engine not built! Use option 1 to build.${NC}"
            exit 1
        fi
    fi
    
    echo -e "${CYAN}Running engine...${NC}"
    $binary
}

# Main menu
echo -e "\n${CYAN}===== simmer BUILD SYSTEM =====${NC}\n"
echo -e "${GREEN}Build Options:${NC}"
echo "  1) Build & Run Engine"
echo "  2) Build with Warning Suppression"
echo -e "${YELLOW}Quick Run (no build):${NC}"
echo "  3) Run Engine"
read -p "Selection [1-5]: " choice

case "$choice" in
    1)
        # Only engine builds get debug/release choice
        echo -e "\n${CYAN}Build Mode:${NC}"
        echo "  1) Debug"
        echo "  2) Release"
        read -p "Selection [1/2]: " build_mode
        BUILD_MODE=$([[ "$build_mode" = "2" ]] && echo "release" || echo "debug")
        
        build_engine
        run_engine
        ;;
    2)
        build_with_warnings_suppressed
        ;;
    3)
        run_engine
        ;;
    *)
        echo -e "${RED}Invalid selection${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Done!${NC}"