# Fish completion for kt command
# This is a static completion script that doesn't require Python startup

# Main commands
complete -c kt -f -n "__fish_use_subcommand" -a "version" -d "Show version information"
complete -c kt -f -n "__fish_use_subcommand" -a "install" -d "Install KTransformers and dependencies"
complete -c kt -f -n "__fish_use_subcommand" -a "update" -d "Update KTransformers to the latest version"
complete -c kt -f -n "__fish_use_subcommand" -a "run" -d "Start model inference server"
complete -c kt -f -n "__fish_use_subcommand" -a "download" -d "Download model weights"
complete -c kt -f -n "__fish_use_subcommand" -a "quant" -d "Quantize model weights"
complete -c kt -f -n "__fish_use_subcommand" -a "bench" -d "Run full benchmark"
complete -c kt -f -n "__fish_use_subcommand" -a "microbench" -d "Run micro-benchmark"
complete -c kt -f -n "__fish_use_subcommand" -a "doctor" -d "Diagnose environment issues"
complete -c kt -f -n "__fish_use_subcommand" -a "config" -d "Manage configuration"
complete -c kt -f -n "__fish_use_subcommand" -a "sft" -d "Fine-tuning with LlamaFactory"

# Global options
complete -c kt -l help -d "Show help message"
complete -c kt -l version -d "Show version"
complete -c kt -l install-completion -d "Install shell completion"
complete -c kt -l show-completion -d "Show completion script"

# Install command options
complete -c kt -f -n "__fish_seen_subcommand_from install" -l source -d "Install from source directory"
complete -c kt -f -n "__fish_seen_subcommand_from install" -l branch -d "Git branch to use"
complete -c kt -f -n "__fish_seen_subcommand_from install" -s y -l yes -d "Skip confirmation prompts"
complete -c kt -f -n "__fish_seen_subcommand_from install" -s f -l force -d "Force reinstall"
complete -c kt -f -n "__fish_seen_subcommand_from install" -l skip-torch -d "Skip PyTorch installation"
complete -c kt -f -n "__fish_seen_subcommand_from install" -s e -l editable -d "Install in editable mode"
complete -c kt -f -n "__fish_seen_subcommand_from install" -l from-source -d "Build from source"
complete -c kt -f -n "__fish_seen_subcommand_from install" -l cpu-instruct -d "CPU instruction set"
complete -c kt -f -n "__fish_seen_subcommand_from install" -l enable-amx -d "Enable Intel AMX"
complete -c kt -f -n "__fish_seen_subcommand_from install" -l disable-amx -d "Disable Intel AMX"
complete -c kt -f -n "__fish_seen_subcommand_from install" -l build-type -d "Build type"
complete -c kt -f -n "__fish_seen_subcommand_from install" -l deps-only -d "Install system dependencies only"
complete -c kt -f -n "__fish_seen_subcommand_from install" -l docker -d "Show Docker installation guide"
complete -c kt -f -n "__fish_seen_subcommand_from install" -l verify -d "Verify installation"
complete -c kt -f -n "__fish_seen_subcommand_from install" -l no-verify -d "Skip verification"
complete -c kt -f -n "__fish_seen_subcommand_from install" -a "inference sft full"

# Update command options
complete -c kt -f -n "__fish_seen_subcommand_from update" -l source -d "Update from source directory"
complete -c kt -f -n "__fish_seen_subcommand_from update" -l pypi -d "Update from PyPI"
complete -c kt -f -n "__fish_seen_subcommand_from update" -s y -l yes -d "Skip confirmation prompts"

# Run command options
complete -c kt -f -n "__fish_seen_subcommand_from run" -l model -d "Model name or path"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l config -d "Config file path"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l host -d "Server host"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l port -d "Server port"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l gpu-experts -d "Number of GPU experts"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l cpu-threads -d "Number of CPU threads"

# Download command options
complete -c kt -f -n "__fish_seen_subcommand_from download" -l output -d "Output directory"
complete -c kt -f -n "__fish_seen_subcommand_from download" -l resume -d "Resume download"
complete -c kt -f -n "__fish_seen_subcommand_from download" -l mirror -d "HuggingFace mirror"

# Quant command options
complete -c kt -f -n "__fish_seen_subcommand_from quant" -l method -d "Quantization method"
complete -c kt -f -n "__fish_seen_subcommand_from quant" -l output -d "Output directory"

# Bench command options
complete -c kt -f -n "__fish_seen_subcommand_from bench microbench" -l model -d "Model name or path"
complete -c kt -f -n "__fish_seen_subcommand_from bench microbench" -l config -d "Config file path"

# Doctor command options
complete -c kt -f -n "__fish_seen_subcommand_from doctor" -l verbose -d "Verbose output"

# Config subcommands
complete -c kt -f -n "__fish_seen_subcommand_from config; and not __fish_seen_subcommand_from show get set reset" -a "show" -d "Show all configuration"
complete -c kt -f -n "__fish_seen_subcommand_from config; and not __fish_seen_subcommand_from show get set reset" -a "get" -d "Get configuration value"
complete -c kt -f -n "__fish_seen_subcommand_from config; and not __fish_seen_subcommand_from show get set reset" -a "set" -d "Set configuration value"
complete -c kt -f -n "__fish_seen_subcommand_from config; and not __fish_seen_subcommand_from show get set reset" -a "reset" -d "Reset to defaults"

# SFT subcommands
complete -c kt -f -n "__fish_seen_subcommand_from sft; and not __fish_seen_subcommand_from train chat export" -a "train" -d "Train model"
complete -c kt -f -n "__fish_seen_subcommand_from sft; and not __fish_seen_subcommand_from train chat export" -a "chat" -d "Chat with model"
complete -c kt -f -n "__fish_seen_subcommand_from sft; and not __fish_seen_subcommand_from train chat export" -a "export" -d "Export model"
