#!/usr/bin/env bash
#
# build-docker-tar.sh - Build Docker image and export to tar file
#
# This script builds a Docker image for ktransformers with standardized naming
# and exports it to a tar file for distribution.
#
# Features:
# - Automatic version detection from built image
# - Standardized naming convention
# - Multi-CPU variant support (AMX/AVX512/AVX2)
# - Configurable build parameters
# - Comprehensive error handling
#
# Usage:
#   ./build-docker-tar.sh [OPTIONS]
#
# Example:
#   ./build-docker-tar.sh \
#     --cuda-version 12.8.1 \
#     --ubuntu-mirror 1 \
#     --http-proxy "http://127.0.0.1:16981" \
#     --output-dir /path/to/output

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
# shellcheck source=docker-utils.sh
source "$SCRIPT_DIR/docker-utils.sh"

################################################################################
# Default Configuration
################################################################################

# Build parameters
CUDA_VERSION="12.8.1"
UBUNTU_MIRROR="0"
HTTP_PROXY=""
HTTPS_PROXY=""
CPU_VARIANT="x86-intel-multi"
FUNCTIONALITY="sft"

# Paths
DOCKERFILE="$SCRIPT_DIR/Dockerfile"
CONTEXT_DIR="$SCRIPT_DIR"
OUTPUT_DIR="."

# Options
DRY_RUN=false
KEEP_IMAGE=false
EXTRA_BUILD_ARGS=()

################################################################################
# Help Message
################################################################################

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Build Docker image and export to tar file with standardized naming.

OPTIONS:
    Build Configuration:
        --cuda-version VERSION      CUDA version (default: 12.8.1)
                                   Examples: 12.8.1, 12.6.1, 13.0.1

        --ubuntu-mirror 0|1         Use Tsinghua mirror for Ubuntu packages
                                   (default: 0)

        --http-proxy URL           HTTP proxy URL
                                   Example: http://127.0.0.1:16981

        --https-proxy URL          HTTPS proxy URL
                                   Example: http://127.0.0.1:16981

        --cpu-variant VARIANT      CPU variant identifier
                                   (default: x86-intel-multi)

        --functionality TYPE       Functionality mode: sft or infer
                                   (default: sft, includes LLaMA-Factory)

    Paths:
        --dockerfile PATH          Path to Dockerfile
                                   (default: ./Dockerfile)

        --context-dir PATH         Docker build context directory
                                   (default: .)

        --output-dir PATH          Output directory for tar file
                                   (default: current directory)

    Options:
        --dry-run                  Preview build command without executing
        --keep-image               Keep Docker image after exporting tar
        --build-arg KEY=VALUE      Additional build arguments (can be repeated)
        -h, --help                 Show this help message

EXAMPLES:
    # Basic build with default settings
    $0

    # Build with CUDA 12.8.1 and mirror
    $0 --cuda-version 12.8.1 --ubuntu-mirror 1

    # Build with proxy and custom output directory
    $0 \\
        --cuda-version 12.8.1 \\
        --http-proxy "http://127.0.0.1:16981" \\
        --https-proxy "http://127.0.0.1:16981" \\
        --output-dir /mnt/data/docker-images

    # Dry run to preview
    $0 --cuda-version 12.8.1 --dry-run

OUTPUT:
    The tar file will be named following the convention:
    sglang-v{ver}_ktransformers-v{ver}_{cpu}_{gpu}_{func}_{timestamp}.tar

    Example: sglang-v0.5.6_ktransformers-v0.4.3_x86-intel-multi_cu128_sft_llamafactory-v0.9.3_20241212143022.tar

EOF
    exit 0
}

################################################################################
# Argument Parsing
################################################################################

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cuda-version)
                CUDA_VERSION="$2"
                shift 2
                ;;
            --ubuntu-mirror)
                UBUNTU_MIRROR="$2"
                shift 2
                ;;
            --http-proxy)
                HTTP_PROXY="$2"
                shift 2
                ;;
            --https-proxy)
                HTTPS_PROXY="$2"
                shift 2
                ;;
            --cpu-variant)
                CPU_VARIANT="$2"
                shift 2
                ;;
            --functionality)
                FUNCTIONALITY="$2"
                shift 2
                ;;
            --dockerfile)
                DOCKERFILE="$2"
                shift 2
                ;;
            --context-dir)
                CONTEXT_DIR="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --keep-image)
                KEEP_IMAGE=true
                shift
                ;;
            --build-arg)
                EXTRA_BUILD_ARGS+=("--build-arg" "$2")
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use -h or --help for usage information"
                exit 1
                ;;
        esac
    done
}

################################################################################
# Validation
################################################################################

validate_config() {
    log_step "Validating configuration"

    # Check Docker is running
    check_docker_running || exit 1

    # Validate CUDA version
    validate_cuda_version "$CUDA_VERSION" || exit 1

    # Check Dockerfile exists
    if [ ! -f "$DOCKERFILE" ]; then
        log_error "Dockerfile not found: $DOCKERFILE"
        exit 1
    fi
    log_info "Using Dockerfile: $DOCKERFILE"

    # Check context directory exists
    if [ ! -d "$CONTEXT_DIR" ]; then
        log_error "Context directory not found: $CONTEXT_DIR"
        exit 1
    fi
    log_info "Using context directory: $CONTEXT_DIR"

    # Create output directory if it doesn't exist
    if [ ! -d "$OUTPUT_DIR" ]; then
        log_info "Creating output directory: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    fi

    # Check output directory is writable
    check_writable "$OUTPUT_DIR" || exit 1
    log_info "Output directory: $OUTPUT_DIR"

    # Check disk space (recommend at least 20GB free)
    check_disk_space 20 "$OUTPUT_DIR" || {
        log_warning "Continuing despite low disk space warning..."
    }

    # Validate functionality mode
    if [[ "$FUNCTIONALITY" != "sft" && "$FUNCTIONALITY" != "infer" ]]; then
        log_error "Invalid functionality mode: $FUNCTIONALITY"
        log_error "Must be 'sft' or 'infer'"
        exit 1
    fi

    log_success "Configuration validated"
}

################################################################################
# Build Docker Image
################################################################################

build_image() {
    local temp_tag="ktransformers:temp-build-$(get_beijing_timestamp)"

    log_step "Building Docker image"
    log_info "Temporary tag: $temp_tag"

    # Prepare build arguments
    local build_args=()
    build_args+=("--build-arg" "CUDA_VERSION=$CUDA_VERSION")
    build_args+=("--build-arg" "UBUNTU_MIRROR=$UBUNTU_MIRROR")
    build_args+=("--build-arg" "CPU_VARIANT=$CPU_VARIANT")
    build_args+=("--build-arg" "BUILD_ALL_CPU_VARIANTS=1")

    # Add proxy settings if provided
    if [ -n "$HTTP_PROXY" ]; then
        build_args+=("--build-arg" "HTTP_PROXY=$HTTP_PROXY")
    fi
    if [ -n "$HTTPS_PROXY" ]; then
        build_args+=("--build-arg" "HTTPS_PROXY=$HTTPS_PROXY")
    fi

    # Add extra build args
    build_args+=("${EXTRA_BUILD_ARGS[@]}")

    # Add network host
    build_args+=("--network" "host")

    # Build command
    local build_cmd=(
        docker build
        -f "$DOCKERFILE"
        "${build_args[@]}"
        -t "$temp_tag"
        "$CONTEXT_DIR"
    )

    # Display build command
    log_info "Build command:"
    echo "  ${build_cmd[*]}"

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Skipping actual build"
        return 0
    fi

    # Execute build
    log_info "Starting Docker build (this may take 30-60 minutes)..."
    if "${build_cmd[@]}"; then
        log_success "Docker image built successfully"
        echo "$temp_tag"
    else
        log_error "Docker build failed"
        exit 1
    fi
}

################################################################################
# Extract Versions and Generate Name
################################################################################

generate_tar_name() {
    local image_tag="$1"
    local timestamp="$2"

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Using placeholder versions"
        # Use placeholder versions for dry run
        local versions="SGLANG_VERSION=0.5.6
KTRANSFORMERS_VERSION=0.4.3
LLAMAFACTORY_VERSION=0.9.3"
    else
        # Extract versions from image
        local versions
        versions=$(extract_versions_from_image "$image_tag")

        if [ $? -ne 0 ]; then
            log_error "Failed to extract versions from image"
            exit 1
        fi

        # Validate versions
        if ! validate_versions "$versions"; then
            log_error "Version validation failed"
            exit 1
        fi
    fi

    # Generate standardized image name
    local tar_name
    tar_name=$(generate_image_name "$versions" "$CUDA_VERSION" "$CPU_VARIANT" "$FUNCTIONALITY" "$timestamp")

    if [ -z "$tar_name" ]; then
        log_error "Failed to generate image name"
        exit 1
    fi

    echo "$tar_name"
}

################################################################################
# Export to Tar
################################################################################

export_to_tar() {
    local image_tag="$1"
    local tar_name="$2"
    local tar_path="$OUTPUT_DIR/${tar_name}.tar"

    log_step "Exporting image to tar file"
    log_info "Output: $tar_path"

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Skipping actual export"
        return 0
    fi

    # Check if tar file already exists
    if [ -f "$tar_path" ]; then
        log_warning "Tar file already exists: $tar_path"
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Export cancelled by user"
            exit 1
        fi
        rm -f "$tar_path"
    fi

    # Export image
    log_info "Exporting image (this may take several minutes)..."
    if docker save -o "$tar_path" "$image_tag"; then
        log_success "Image exported successfully"

        # Get file size
        local size
        size=$(du -h "$tar_path" | cut -f1)
        log_info "Tar file size: $size"
    else
        log_error "Failed to export image"
        exit 1
    fi

    echo "$tar_path"
}

################################################################################
# Cleanup
################################################################################

cleanup() {
    local image_tag="$1"

    if [ "$KEEP_IMAGE" = true ]; then
        log_info "Keeping Docker image as requested: $image_tag"
    else
        cleanup_temp_images "$image_tag"
    fi
}

################################################################################
# Main
################################################################################

main() {
    log_step "KTransformers Docker Image Build and Export"

    # Parse arguments
    parse_args "$@"

    # Validate configuration
    validate_config

    # Generate timestamp
    TIMESTAMP=$(get_beijing_timestamp)
    log_info "Build timestamp: $TIMESTAMP"

    # Display configuration
    display_summary "Build Configuration" \
        "CUDA Version: $CUDA_VERSION" \
        "Ubuntu Mirror: $UBUNTU_MIRROR" \
        "CPU Variant: $CPU_VARIANT" \
        "Functionality: $FUNCTIONALITY" \
        "HTTP Proxy: ${HTTP_PROXY:-<not set>}" \
        "HTTPS Proxy: ${HTTPS_PROXY:-<not set>}" \
        "Dockerfile: $DOCKERFILE" \
        "Context Dir: $CONTEXT_DIR" \
        "Output Dir: $OUTPUT_DIR" \
        "Timestamp: $TIMESTAMP" \
        "Dry Run: $DRY_RUN"

    # Build image
    TEMP_TAG=$(build_image)

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Preview complete"
        exit 0
    fi

    # Generate tar name
    TAR_NAME=$(generate_tar_name "$TEMP_TAG" "$TIMESTAMP")
    log_info "Generated tar name: $TAR_NAME.tar"

    # Export to tar
    TAR_PATH=$(export_to_tar "$TEMP_TAG" "$TAR_NAME")

    # Cleanup
    cleanup "$TEMP_TAG"

    # Display summary
    display_summary "Build Complete" \
        "Docker Image: $TEMP_TAG ($([ "$KEEP_IMAGE" = true ] && echo "kept" || echo "removed"))" \
        "Tar File: $TAR_PATH" \
        "" \
        "To load the image:" \
        "  docker load -i $TAR_PATH" \
        "" \
        "To run the container:" \
        "  docker run -it --rm ${TAR_NAME} /bin/bash"

    log_success "All done!"
}

# Run main function
main "$@"
