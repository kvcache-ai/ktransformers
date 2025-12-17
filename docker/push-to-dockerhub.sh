#!/usr/bin/env bash
#
# push-to-dockerhub.sh - Build and push Docker image to DockerHub
#
# This script builds a Docker image for ktransformers with standardized naming
# and pushes it to DockerHub with both full and simplified tags.
#
# Features:
# - Automatic version detection
# - Standardized naming convention
# - Multi-CPU variant support (AMX/AVX512/AVX2)
# - Full and simplified tag support
# - Retry logic for network failures
# - Comprehensive error handling
#
# Usage:
#   ./push-to-dockerhub.sh [OPTIONS]
#
# Example:
#   ./push-to-dockerhub.sh \
#     --cuda-version 12.8.1 \
#     --repository kvcache/ktransformers \
#     --also-push-simplified

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

# Registry settings
REGISTRY="docker.io"
REPOSITORY=""  # Must be provided by user

# Options
DRY_RUN=false
SKIP_BUILD=false
ALSO_PUSH_SIMPLIFIED=false
MAX_RETRIES=3
RETRY_DELAY=5
EXTRA_BUILD_ARGS=()

################################################################################
# Help Message
################################################################################

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Build and push Docker image to DockerHub with standardized naming.

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

    Registry Settings:
        --registry REGISTRY        Docker registry (default: docker.io)
                                   Examples: docker.io, ghcr.io

        --repository REPO          Repository name (REQUIRED)
                                   Example: kvcache/ktransformers

    Options:
        --skip-build               Skip build if image exists locally
        --also-push-simplified     Also push simplified tag (v{ver}-{cuda})
        --max-retries N            Maximum push retries (default: 3)
        --retry-delay SECONDS      Delay between retries (default: 5)
        --dry-run                  Preview commands without executing
        --build-arg KEY=VALUE      Additional build arguments (can be repeated)
        -h, --help                 Show this help message

EXAMPLES:
    # Basic push
    $0 --repository kvcache/ktransformers

    # Push with simplified tag
    $0 \\
        --repository kvcache/ktransformers \\
        --cuda-version 12.8.1 \\
        --also-push-simplified

    # Skip build if image exists
    $0 \\
        --repository kvcache/ktransformers \\
        --skip-build

    # Dry run to preview
    $0 --repository kvcache/ktransformers --dry-run

OUTPUT:
    The image will be pushed with tags:

    Full tag:
      {registry}/{repository}:sglang-v{ver}_ktransformers-v{ver}_{cpu}_{gpu}_{func}_{timestamp}

    Example:
      docker.io/kvcache/ktransformers:sglang-v0.5.6_ktransformers-v0.4.3_x86-intel-multi_cu128_sft_llamafactory-v0.9.3_20241212143022

    Simplified tag (if --also-push-simplified):
      {registry}/{repository}:v{ktransformers-ver}-{cuda}

    Example:
      docker.io/kvcache/ktransformers:v0.4.3-cu128

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
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --repository)
                REPOSITORY="$2"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --also-push-simplified)
                ALSO_PUSH_SIMPLIFIED=true
                shift
                ;;
            --max-retries)
                MAX_RETRIES="$2"
                shift 2
                ;;
            --retry-delay)
                RETRY_DELAY="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
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

    # Check Docker login
    check_docker_login "$REGISTRY" || exit 1

    # Validate CUDA version
    validate_cuda_version "$CUDA_VERSION" || exit 1

    # Check repository is provided
    if [ -z "$REPOSITORY" ]; then
        log_error "Repository name is required"
        log_error "Use --repository to specify (e.g., kvcache/ktransformers)"
        exit 1
    fi
    log_info "Target repository: $REGISTRY/$REPOSITORY"

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
    local temp_tag="ktransformers:temp-push-$(get_beijing_timestamp)"

    # Check if we should skip build
    if [ "$SKIP_BUILD" = true ]; then
        log_info "Checking for existing local image..." >&2
        # Try to find an existing image
        # This is a best-effort search for recent builds
        local existing_image
        existing_image=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "ktransformers:temp-" | head -1 || echo "")

        if [ -n "$existing_image" ]; then
            log_info "Found existing image: $existing_image" >&2
            echo "$existing_image"
            return 0
        else
            log_warning "No existing image found, will build" >&2
        fi
    fi

    log_step "Building Docker image" >&2
    log_info "Temporary tag: $temp_tag" >&2

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
    {
        log_info "Build command:"
        echo "  ${build_cmd[*]}"
    } >&2

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Skipping actual build" >&2
        return 0
    fi

    # Execute build
    log_info "Starting Docker build (this may take 30-60 minutes)..." >&2
    if "${build_cmd[@]}" >&2; then
        log_success "Docker image built successfully" >&2
        echo "$temp_tag"
    else
        log_error "Docker build failed" >&2
        exit 1
    fi
}

################################################################################
# Generate Tags
################################################################################

generate_tags() {
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

    # Generate full tag
    local full_tag
    full_tag=$(generate_image_name "$versions" "$CUDA_VERSION" "$CPU_VARIANT" "$FUNCTIONALITY" "$timestamp")

    if [ -z "$full_tag" ]; then
        log_error "Failed to generate image name"
        exit 1
    fi

    echo "FULL_TAG=$full_tag"

    # Generate simplified tag if requested
    if [ "$ALSO_PUSH_SIMPLIFIED" = true ]; then
        local ktrans_ver
        ktrans_ver=$(echo "$versions" | grep "^KTRANSFORMERS_VERSION=" | cut -d= -f2)

        local simplified_tag
        simplified_tag=$(generate_simplified_tag "$ktrans_ver" "$CUDA_VERSION")

        echo "SIMPLIFIED_TAG=$simplified_tag"
    fi
}

################################################################################
# Push to Registry
################################################################################

push_image_with_retry() {
    local source_tag="$1"
    local target_tag="$2"
    local attempt=1

    log_step "Pushing image: $target_tag"

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Skipping actual push"
        log_info "Would execute:"
        echo "  docker tag $source_tag $target_tag"
        echo "  docker push $target_tag"
        return 0
    fi

    # Tag the image
    log_info "Tagging image..."
    if ! docker tag "$source_tag" "$target_tag"; then
        log_error "Failed to tag image"
        return 1
    fi

    # Push with retry logic
    while [ $attempt -le "$MAX_RETRIES" ]; do
        log_info "Push attempt $attempt/$MAX_RETRIES..."

        if docker push "$target_tag"; then
            log_success "Successfully pushed: $target_tag"
            return 0
        else
            log_warning "Push failed (attempt $attempt/$MAX_RETRIES)"

            if [ $attempt -lt "$MAX_RETRIES" ]; then
                log_info "Retrying in ${RETRY_DELAY} seconds..."
                sleep "$RETRY_DELAY"
            fi

            ((attempt++))
        fi
    done

    log_error "Failed to push after $MAX_RETRIES attempts"
    return 1
}

################################################################################
# Main
################################################################################

main() {
    log_step "KTransformers Docker Image Build and Push"

    # Parse arguments
    parse_args "$@"

    # Validate configuration
    validate_config

    # Generate timestamp
    TIMESTAMP=$(get_beijing_timestamp)
    log_info "Build timestamp: $TIMESTAMP"

    # Display configuration
    display_summary "Push Configuration" \
        "CUDA Version: $CUDA_VERSION" \
        "Ubuntu Mirror: $UBUNTU_MIRROR" \
        "CPU Variant: $CPU_VARIANT" \
        "Functionality: $FUNCTIONALITY" \
        "Registry: $REGISTRY" \
        "Repository: $REPOSITORY" \
        "Push Simplified: $ALSO_PUSH_SIMPLIFIED" \
        "Skip Build: $SKIP_BUILD" \
        "HTTP Proxy: ${HTTP_PROXY:-<not set>}" \
        "HTTPS Proxy: ${HTTPS_PROXY:-<not set>}" \
        "Dockerfile: $DOCKERFILE" \
        "Context Dir: $CONTEXT_DIR" \
        "Timestamp: $TIMESTAMP" \
        "Dry Run: $DRY_RUN"

    # Build image
    TEMP_TAG=$(build_image)

    if [ "$DRY_RUN" = true ]; then
        TEMP_TAG="ktransformers:temp-dryrun"
    fi

    # Generate tags
    log_step "Generating tags"
    TAG_INFO=$(generate_tags "$TEMP_TAG" "$TIMESTAMP")

    # Parse tag info
    FULL_TAG=$(echo "$TAG_INFO" | grep "^FULL_TAG=" | cut -d= -f2)
    SIMPLIFIED_TAG=$(echo "$TAG_INFO" | grep "^SIMPLIFIED_TAG=" | cut -d= -f2 || echo "")

    log_info "Full tag: $FULL_TAG"
    if [ -n "$SIMPLIFIED_TAG" ]; then
        log_info "Simplified tag: $SIMPLIFIED_TAG"
    fi

    # Push full tag
    FULL_IMAGE="$REGISTRY/$REPOSITORY:$FULL_TAG"
    if ! push_image_with_retry "$TEMP_TAG" "$FULL_IMAGE"; then
        log_error "Failed to push full tag"
        exit 1
    fi

    # Push simplified tag if requested
    if [ -n "$SIMPLIFIED_TAG" ]; then
        SIMPLIFIED_IMAGE="$REGISTRY/$REPOSITORY:$SIMPLIFIED_TAG"
        if ! push_image_with_retry "$TEMP_TAG" "$SIMPLIFIED_IMAGE"; then
            log_warning "Failed to push simplified tag, but continuing..."
        fi
    fi

    # Cleanup temporary image
    if [ "$DRY_RUN" = false ]; then
        log_step "Cleaning up temporary image"
        cleanup_temp_images "$TEMP_TAG"
    fi

    # Display summary
    local summary_lines=(
        "Successfully pushed images:"
        ""
        "Full tag:"
        "  $FULL_IMAGE"
        ""
    )

    if [ -n "$SIMPLIFIED_TAG" ]; then
        summary_lines+=(
            "Simplified tag:"
            "  $SIMPLIFIED_IMAGE"
            ""
        )
    fi

    summary_lines+=(
        "To pull the image:"
        "  docker pull $FULL_IMAGE"
        ""
        "To run the container:"
        "  docker run -it --rm $FULL_IMAGE /bin/bash"
    )

    display_summary "Push Complete" "${summary_lines[@]}"

    log_success "All done!"
}

# Run main function
main "$@"
#!/usr/bin/env bash
#
# push-to-dockerhub.sh - Build and push Docker image to DockerHub
#
# This script builds a Docker image for ktransformers with standardized naming
# and pushes it to DockerHub with both full and simplified tags.
#
# Features:
# - Automatic version detection
# - Standardized naming convention
# - Multi-CPU variant support (AMX/AVX512/AVX2)
# - Full and simplified tag support
# - Retry logic for network failures
# - Comprehensive error handling
#
# Usage:
#   ./push-to-dockerhub.sh [OPTIONS]
#
# Example:
#   ./push-to-dockerhub.sh \
#     --cuda-version 12.8.1 \
#     --repository kvcache/ktransformers \
#     --also-push-simplified

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

# Registry settings
REGISTRY="docker.io"
REPOSITORY=""  # Must be provided by user

# Options
DRY_RUN=false
SKIP_BUILD=false
ALSO_PUSH_SIMPLIFIED=false
MAX_RETRIES=3
RETRY_DELAY=5
EXTRA_BUILD_ARGS=()

################################################################################
# Help Message
################################################################################

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Build and push Docker image to DockerHub with standardized naming.

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

    Registry Settings:
        --registry REGISTRY        Docker registry (default: docker.io)
                                   Examples: docker.io, ghcr.io

        --repository REPO          Repository name (REQUIRED)
                                   Example: kvcache/ktransformers

    Options:
        --skip-build               Skip build if image exists locally
        --also-push-simplified     Also push simplified tag (v{ver}-{cuda})
        --max-retries N            Maximum push retries (default: 3)
        --retry-delay SECONDS      Delay between retries (default: 5)
        --dry-run                  Preview commands without executing
        --build-arg KEY=VALUE      Additional build arguments (can be repeated)
        -h, --help                 Show this help message

EXAMPLES:
    # Basic push
    $0 --repository kvcache/ktransformers

    # Push with simplified tag
    $0 \\
        --repository kvcache/ktransformers \\
        --cuda-version 12.8.1 \\
        --also-push-simplified

    # Skip build if image exists
    $0 \\
        --repository kvcache/ktransformers \\
        --skip-build

    # Dry run to preview
    $0 --repository kvcache/ktransformers --dry-run

OUTPUT:
    The image will be pushed with tags:

    Full tag:
      {registry}/{repository}:sglang-v{ver}_ktransformers-v{ver}_{cpu}_{gpu}_{func}_{timestamp}

    Example:
      docker.io/kvcache/ktransformers:sglang-v0.5.6_ktransformers-v0.4.3_x86-intel-multi_cu128_sft_llamafactory-v0.9.3_20241212143022

    Simplified tag (if --also-push-simplified):
      {registry}/{repository}:v{ktransformers-ver}-{cuda}

    Example:
      docker.io/kvcache/ktransformers:v0.4.3-cu128

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
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --repository)
                REPOSITORY="$2"
                shift 2
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --also-push-simplified)
                ALSO_PUSH_SIMPLIFIED=true
                shift
                ;;
            --max-retries)
                MAX_RETRIES="$2"
                shift 2
                ;;
            --retry-delay)
                RETRY_DELAY="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
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

    # Check Docker login
    check_docker_login "$REGISTRY" || exit 1

    # Validate CUDA version
    validate_cuda_version "$CUDA_VERSION" || exit 1

    # Check repository is provided
    if [ -z "$REPOSITORY" ]; then
        log_error "Repository name is required"
        log_error "Use --repository to specify (e.g., kvcache/ktransformers)"
        exit 1
    fi
    log_info "Target repository: $REGISTRY/$REPOSITORY"

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
    local temp_tag="ktransformers:temp-push-$(get_beijing_timestamp)"

    # Check if we should skip build
    if [ "$SKIP_BUILD" = true ]; then
        log_info "Checking for existing local image..." >&2
        # Try to find an existing image
        # This is a best-effort search for recent builds
        local existing_image
        existing_image=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "ktransformers:temp-" | head -1 || echo "")

        if [ -n "$existing_image" ]; then
            log_info "Found existing image: $existing_image" >&2
            echo "$existing_image"
            return 0
        else
            log_warning "No existing image found, will build" >&2
        fi
    fi

    log_step "Building Docker image" >&2
    log_info "Temporary tag: $temp_tag" >&2

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
    {
        log_info "Build command:"
        echo "  ${build_cmd[*]}"
    } >&2

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Skipping actual build" >&2
        return 0
    fi

    # Execute build
    log_info "Starting Docker build (this may take 30-60 minutes)..." >&2
    if "${build_cmd[@]}" >&2; then
        log_success "Docker image built successfully" >&2
        echo "$temp_tag"
    else
        log_error "Docker build failed" >&2
        exit 1
    fi
}

################################################################################
# Generate Tags
################################################################################

generate_tags() {
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

    # Generate full tag
    local full_tag
    full_tag=$(generate_image_name "$versions" "$CUDA_VERSION" "$CPU_VARIANT" "$FUNCTIONALITY" "$timestamp")

    if [ -z "$full_tag" ]; then
        log_error "Failed to generate image name"
        exit 1
    fi

    echo "FULL_TAG=$full_tag"

    # Generate simplified tag if requested
    if [ "$ALSO_PUSH_SIMPLIFIED" = true ]; then
        local ktrans_ver
        ktrans_ver=$(echo "$versions" | grep "^KTRANSFORMERS_VERSION=" | cut -d= -f2)

        local simplified_tag
        simplified_tag=$(generate_simplified_tag "$ktrans_ver" "$CUDA_VERSION")

        echo "SIMPLIFIED_TAG=$simplified_tag"
    fi
}

################################################################################
# Push to Registry
################################################################################

push_image_with_retry() {
    local source_tag="$1"
    local target_tag="$2"
    local attempt=1

    log_step "Pushing image: $target_tag"

    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN: Skipping actual push"
        log_info "Would execute:"
        echo "  docker tag $source_tag $target_tag"
        echo "  docker push $target_tag"
        return 0
    fi

    # Tag the image
    log_info "Tagging image..."
    if ! docker tag "$source_tag" "$target_tag"; then
        log_error "Failed to tag image"
        return 1
    fi

    # Push with retry logic
    while [ $attempt -le "$MAX_RETRIES" ]; do
        log_info "Push attempt $attempt/$MAX_RETRIES..."

        if docker push "$target_tag"; then
            log_success "Successfully pushed: $target_tag"
            return 0
        else
            log_warning "Push failed (attempt $attempt/$MAX_RETRIES)"

            if [ $attempt -lt "$MAX_RETRIES" ]; then
                log_info "Retrying in ${RETRY_DELAY} seconds..."
                sleep "$RETRY_DELAY"
            fi

            ((attempt++))
        fi
    done

    log_error "Failed to push after $MAX_RETRIES attempts"
    return 1
}

################################################################################
# Main
################################################################################

main() {
    log_step "KTransformers Docker Image Build and Push"

    # Parse arguments
    parse_args "$@"

    # Validate configuration
    validate_config

    # Generate timestamp
    TIMESTAMP=$(get_beijing_timestamp)
    log_info "Build timestamp: $TIMESTAMP"

    # Display configuration
    display_summary "Push Configuration" \
        "CUDA Version: $CUDA_VERSION" \
        "Ubuntu Mirror: $UBUNTU_MIRROR" \
        "CPU Variant: $CPU_VARIANT" \
        "Functionality: $FUNCTIONALITY" \
        "Registry: $REGISTRY" \
        "Repository: $REPOSITORY" \
        "Push Simplified: $ALSO_PUSH_SIMPLIFIED" \
        "Skip Build: $SKIP_BUILD" \
        "HTTP Proxy: ${HTTP_PROXY:-<not set>}" \
        "HTTPS Proxy: ${HTTPS_PROXY:-<not set>}" \
        "Dockerfile: $DOCKERFILE" \
        "Context Dir: $CONTEXT_DIR" \
        "Timestamp: $TIMESTAMP" \
        "Dry Run: $DRY_RUN"

    # Build image
    TEMP_TAG=$(build_image)

    if [ "$DRY_RUN" = true ]; then
        TEMP_TAG="ktransformers:temp-dryrun"
    fi

    # Generate tags
    log_step "Generating tags"
    TAG_INFO=$(generate_tags "$TEMP_TAG" "$TIMESTAMP")

    # Parse tag info
    FULL_TAG=$(echo "$TAG_INFO" | grep "^FULL_TAG=" | cut -d= -f2)
    SIMPLIFIED_TAG=$(echo "$TAG_INFO" | grep "^SIMPLIFIED_TAG=" | cut -d= -f2 || echo "")

    log_info "Full tag: $FULL_TAG"
    if [ -n "$SIMPLIFIED_TAG" ]; then
        log_info "Simplified tag: $SIMPLIFIED_TAG"
    fi

    # Push full tag
    FULL_IMAGE="$REGISTRY/$REPOSITORY:$FULL_TAG"
    if ! push_image_with_retry "$TEMP_TAG" "$FULL_IMAGE"; then
        log_error "Failed to push full tag"
        exit 1
    fi

    # Push simplified tag if requested
    if [ -n "$SIMPLIFIED_TAG" ]; then
        SIMPLIFIED_IMAGE="$REGISTRY/$REPOSITORY:$SIMPLIFIED_TAG"
        if ! push_image_with_retry "$TEMP_TAG" "$SIMPLIFIED_IMAGE"; then
            log_warning "Failed to push simplified tag, but continuing..."
        fi
    fi

    # Cleanup temporary image
    if [ "$DRY_RUN" = false ]; then
        log_step "Cleaning up temporary image"
        cleanup_temp_images "$TEMP_TAG"
    fi

    # Display summary
    local summary_lines=(
        "Successfully pushed images:"
        ""
        "Full tag:"
        "  $FULL_IMAGE"
        ""
    )

    if [ -n "$SIMPLIFIED_TAG" ]; then
        summary_lines+=(
            "Simplified tag:"
            "  $SIMPLIFIED_IMAGE"
            ""
        )
    fi

    summary_lines+=(
        "To pull the image:"
        "  docker pull $FULL_IMAGE"
        ""
        "To run the container:"
        "  docker run -it --rm $FULL_IMAGE /bin/bash"
    )

    display_summary "Push Complete" "${summary_lines[@]}"

    log_success "All done!"
}

# Run main function
main "$@"
