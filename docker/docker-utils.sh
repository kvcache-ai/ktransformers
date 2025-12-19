#!/usr/bin/env bash
#
# docker-utils.sh - Shared utility functions for Docker image build and publish scripts
#
# This script provides common functions for:
# - Timestamp generation (Beijing timezone)
# - Version extraction from Docker images
# - Image name generation following naming conventions
# - Colored logging
# - Validation and error handling
#
# Usage: source docker-utils.sh

set -euo pipefail

# Color codes for logging
COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_CYAN='\033[0;36m'
COLOR_RESET='\033[0m'

################################################################################
# Logging Functions
################################################################################

log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $*"
}

log_success() {
    echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $*"
}

log_warning() {
    echo -e "${COLOR_YELLOW}[WARNING]${COLOR_RESET} $*"
}

log_error() {
    echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $*" >&2
}

log_step() {
    echo -e "\n${COLOR_CYAN}==>${COLOR_RESET} $*"
}

################################################################################
# Timestamp Functions
################################################################################

# Generate timestamp in Beijing timezone (UTC+8)
# Format: YYYYMMDDHHMMSS
# Example: 20241212143022
get_beijing_timestamp() {
    # Try to use TZ environment variable approach
    if date --version &>/dev/null 2>&1; then
        # GNU date (Linux)
        TZ='Asia/Shanghai' date '+%Y%m%d%H%M%S'
    else
        # BSD date (macOS)
        TZ='Asia/Shanghai' date '+%Y%m%d%H%M%S'
    fi
}

################################################################################
# CUDA Version Parsing
################################################################################

# Parse CUDA version to short format
# Input: 12.8.1 or 12.8 or 13.0.1
# Output: cu128 or cu130
parse_cuda_short_version() {
    local cuda_version="$1"

    # Extract major and minor version
    local major minor
    major=$(echo "$cuda_version" | cut -d. -f1)
    minor=$(echo "$cuda_version" | cut -d. -f2)

    # Validate
    if [[ ! "$major" =~ ^[0-9]+$ ]] || [[ ! "$minor" =~ ^[0-9]+$ ]]; then
        log_error "Invalid CUDA version format: $cuda_version"
        log_error "Expected format: X.Y.Z (e.g., 12.8.1)"
        return 1
    fi

    echo "cu${major}${minor}"
}

################################################################################
# Version Extraction
################################################################################

# Extract versions from built Docker image
# Input: image tag (e.g., ktransformers:temp-build-20241212)
# Output: Sets environment variables or prints to stdout
#   SGLANG_VERSION=x.y.z
#   KTRANSFORMERS_VERSION=x.y.z
#   LLAMAFACTORY_VERSION=x.y.z
extract_versions_from_image() {
    local image_tag="$1"

    log_step "Extracting versions from image: $image_tag"

    # Check if image exists
    if ! docker image inspect "$image_tag" &>/dev/null; then
        log_error "Image not found: $image_tag"
        return 1
    fi

    # Extract versions.env file from the image
    local versions_content
    versions_content=$(docker run --rm "$image_tag" cat /workspace/versions.env 2>/dev/null)

    if [ -z "$versions_content" ]; then
        log_error "Failed to extract versions from image"
        log_error "The /workspace/versions.env file may not exist in the image"
        return 1
    fi

    # Parse and display versions
    log_info "Extracted versions:"
    echo "$versions_content" | while IFS= read -r line; do
        log_info "  $line"
    done

    # Output the content (caller can parse this or eval it)
    echo "$versions_content"
}

# Validate that all required versions were extracted
# Input: versions string (output from extract_versions_from_image)
validate_versions() {
    local versions="$1"
    local all_valid=true

    # Check each required version
    for var in SGLANG_VERSION KTRANSFORMERS_VERSION LLAMAFACTORY_VERSION; do
        local value
        value=$(echo "$versions" | grep "^${var}=" | cut -d= -f2)

        if [ -z "$value" ]; then
            log_error "Missing version: $var"
            all_valid=false
        elif [ "$value" = "unknown" ]; then
            log_warning "Version is 'unknown': $var"
            # Don't fail, but warn user
        fi
    done

    if [ "$all_valid" = false ]; then
        return 1
    fi

    return 0
}

################################################################################
# Image Naming
################################################################################

# Generate standardized image name
# Input:
#   $1: versions string (from extract_versions_from_image)
#   $2: cuda_version (e.g., 12.8.1)
#   $3: cpu_variant (e.g., x86-intel-multi)
#   $4: functionality (e.g., sft_llamafactory or infer)
#   $5: timestamp (optional, will generate if not provided)
# Output: Standardized image name
# Format: sglang-v{ver}_ktransformers-v{ver}_{cpu}_{gpu}_{func}_{timestamp}
generate_image_name() {
    local versions="$1"
    local cuda_version="$2"
    local cpu_variant="$3"
    local functionality="$4"
    local timestamp="${5:-$(get_beijing_timestamp)}"

    # Parse versions from the versions string
    local sglang_ver ktrans_ver llama_ver
    sglang_ver=$(echo "$versions" | grep "^SGLANG_VERSION=" | cut -d= -f2)
    ktrans_ver=$(echo "$versions" | grep "^KTRANSFORMERS_VERSION=" | cut -d= -f2)
    llama_ver=$(echo "$versions" | grep "^LLAMAFACTORY_VERSION=" | cut -d= -f2)

    # Validate versions were extracted
    if [ -z "$sglang_ver" ] || [ -z "$ktrans_ver" ] || [ -z "$llama_ver" ]; then
        log_error "Failed to parse versions from input"
        return 1
    fi

    # Parse CUDA short version
    local cuda_short
    cuda_short=$(parse_cuda_short_version "$cuda_version")

    # Build functionality string
    local func_str
    if [ "$functionality" = "sft" ]; then
        func_str="sft_llamafactory-v${llama_ver}"
    else
        func_str="infer"
    fi

    # Generate full image name
    # Format: sglang-v{ver}_ktransformers-v{ver}_{cpu}_{gpu}_{func}_{timestamp}
    local image_name
    image_name="sglang-v${sglang_ver}_ktransformers-v${ktrans_ver}_${cpu_variant}_${cuda_short}_${func_str}_${timestamp}"

    echo "$image_name"
}

# Generate simplified tag for DockerHub
# Input:
#   $1: ktransformers_version (e.g., 0.4.3)
#   $2: cuda_version (e.g., 12.8.1)
# Output: Simplified tag (e.g., v0.4.3-cu128)
generate_simplified_tag() {
    local ktrans_ver="$1"
    local cuda_version="$2"

    local cuda_short
    cuda_short=$(parse_cuda_short_version "$cuda_version")

    echo "v${ktrans_ver}-${cuda_short}"
}

################################################################################
# Validation Functions
################################################################################

# Check if Docker daemon is running
check_docker_running() {
    if ! docker info &>/dev/null; then
        log_error "Docker daemon is not running"
        log_error "Please start Docker and try again"
        return 1
    fi
    return 0
}

# Check if user is logged into Docker registry
# Input: registry (optional, default: docker.io)
check_docker_login() {
    local registry="${1:-docker.io}"

    # Try to check auth by attempting a trivial operation
    if ! docker login --help &>/dev/null; then
        log_error "Docker CLI is not available"
        return 1
    fi

    # Note: This is a best-effort check
    # docker login status is not always easy to check programmatically
    log_info "Assuming Docker login is configured"
    log_info "If push fails, please run: docker login $registry"

    return 0
}

# Validate CUDA version format
validate_cuda_version() {
    local cuda_version="$1"

    if [[ ! "$cuda_version" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
        log_error "Invalid CUDA version format: $cuda_version"
        log_error "Expected format: X.Y or X.Y.Z (e.g., 12.8 or 12.8.1)"
        return 1
    fi

    return 0
}

# Check available disk space
# Input: required space in GB
check_disk_space() {
    local required_gb="$1"
    local output_dir="${2:-.}"

    # Get available space in GB (works on Linux and macOS)
    local available_kb
    if df -k "$output_dir" &>/dev/null; then
        available_kb=$(df -k "$output_dir" | tail -1 | awk '{print $4}')
        local available_gb=$((available_kb / 1024 / 1024))

        log_info "Available disk space: ${available_gb}GB"

        if [ "$available_gb" -lt "$required_gb" ]; then
            log_warning "Low disk space: ${available_gb}GB available, ${required_gb}GB recommended"
            return 1
        fi
    else
        log_warning "Unable to check disk space"
    fi

    return 0
}

# Check if file/directory exists and is writable
check_writable() {
    local path="$1"

    if [ -e "$path" ]; then
        if [ ! -w "$path" ]; then
            log_error "Path exists but is not writable: $path"
            return 1
        fi
    else
        # Try to create parent directory to test writability
        local parent_dir
        parent_dir=$(dirname "$path")
        if [ ! -w "$parent_dir" ]; then
            log_error "Parent directory is not writable: $parent_dir"
            return 1
        fi
    fi

    return 0
}

################################################################################
# Cleanup Functions
################################################################################

# Remove intermediate Docker images
cleanup_temp_images() {
    local image_tag="$1"

    log_step "Cleaning up temporary image: $image_tag"

    if docker image inspect "$image_tag" &>/dev/null; then
        docker rmi "$image_tag" &>/dev/null || true
        log_success "Cleaned up temporary image"
    fi
}

################################################################################
# Display Functions
################################################################################

# Display a summary box
display_summary() {
    local title="$1"
    shift
    local lines=("$@")

    local width=80
    local border=$(printf '=%.0s' $(seq 1 $width))

    echo ""
    echo "$border"
    echo "  $title"
    echo "$border"
    for line in "${lines[@]}"; do
        echo "  $line"
    done
    echo "$border"
    echo ""
}

################################################################################
# Export functions
################################################################################

# Export all functions so they can be used by scripts that source this file
export -f log_info log_success log_warning log_error log_step
export -f get_beijing_timestamp
export -f parse_cuda_short_version
export -f extract_versions_from_image validate_versions
export -f generate_image_name generate_simplified_tag
export -f check_docker_running check_docker_login validate_cuda_version
export -f check_disk_space check_writable
export -f cleanup_temp_images
export -f display_summary
