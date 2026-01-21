#!/bin/bash

set -e

PLATFORM="linux/amd64"
IMAGE_NAME="circt-vuln-cve-pending"

echo "🚀 CIRCT 漏洞复现环境快速启动脚本"
echo "   CIRCT Vulnerability Reproduction Quick Start"
echo ""
echo "Image: $IMAGE_NAME"
echo "Platform: $PLATFORM"
echo ""
echo "=============================================="
echo ""

case "${1:-run}" in
    build)
        echo "📦 Building Docker image..."
        docker build --platform "$PLATFORM" -t "$IMAGE_NAME" .
        echo ""
        echo "✅ Build completed!"
        echo "Run './test.sh run' to test the vulnerability"
        ;;
    
    run)
        echo "🧪 Running vulnerability reproduction..."
        echo ""
        docker run --platform "$PLATFORM" --rm "$IMAGE_NAME"
        ;;
    
    save)
        echo "💾 Running and saving output files to ./results/..."
        mkdir -p results
        docker run --platform "$PLATFORM" --rm -v "$(pwd)/results:/vuln-reproduction/output" "$IMAGE_NAME"
        echo ""
        echo "✅ Output files saved in ./results/"
        ls -lh results/
        ;;
    
    vuln)
        echo "🔴 Testing VULNERABLE code only..."
        docker run --platform "$PLATFORM" --rm "$IMAGE_NAME" --vuln-only
        ;;
    
    workaround)
        echo "🟢 Testing WORKAROUND code only..."
        docker run --platform "$PLATFORM" --rm "$IMAGE_NAME" --workaround-only
        ;;
    
    analyze)
        echo "🔍 Running IR analysis..."
        mkdir -p results
        docker run --platform "$PLATFORM" --rm -v "$(pwd)/results:/vuln-reproduction/output" "$IMAGE_NAME" --analyze
        echo ""
        echo "✅ IR dumps saved in ./results/"
        ls -lh results/*.mlir 2>/dev/null || echo "No IR files generated"
        ;;
    
    shell)
        echo "💻 Opening interactive shell in container..."
        docker run --platform "$PLATFORM" --rm -it --entrypoint /bin/bash "$IMAGE_NAME"
        ;;
    
    clean)
        echo "🧹 Cleaning up..."
        docker rmi "$IMAGE_NAME" 2>/dev/null || true
        rm -rf results/
        echo "✅ Cleanup completed"
        ;;
    
    help|--help|-h|*)
        echo "Usage: ./test.sh [command]"
        echo ""
        echo "Commands:"
        echo "  build       - Build Docker image"
        echo "  run         - Run full vulnerability reproduction (default)"
        echo "  save        - Run and save output files to ./results/"
        echo "  vuln        - Test vulnerable code only"
        echo "  workaround  - Test workaround code only"
        echo "  analyze     - Run IR analysis and save dumps"
        echo "  shell       - Open interactive shell in container"
        echo "  clean       - Remove Docker image and results"
        echo "  help        - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./test.sh build        # Build the image first"
        echo "  ./test.sh run          # Run full test"
        echo "  ./test.sh save         # Save output files"
        echo "  ./test.sh shell        # Debug in container"
        echo ""
        ;;
esac
