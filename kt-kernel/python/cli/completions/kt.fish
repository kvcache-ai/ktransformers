# Fish completion for kt command
# This is a static completion script that doesn't require Python startup

# Main commands
complete -c kt -f -n "__fish_use_subcommand" -a "version" -d "Show version information"
complete -c kt -f -n "__fish_use_subcommand" -a "run" -d "Start model inference server"
complete -c kt -f -n "__fish_use_subcommand" -a "chat" -d "Interactive chat with running model"
complete -c kt -f -n "__fish_use_subcommand" -a "quant" -d "Quantize model weights"
complete -c kt -f -n "__fish_use_subcommand" -a "bench" -d "Run full benchmark"
complete -c kt -f -n "__fish_use_subcommand" -a "microbench" -d "Run micro-benchmark"
complete -c kt -f -n "__fish_use_subcommand" -a "doctor" -d "Diagnose environment issues"
complete -c kt -f -n "__fish_use_subcommand" -a "model" -d "Manage models and storage paths"
complete -c kt -f -n "__fish_use_subcommand" -a "config" -d "Manage configuration"
complete -c kt -f -n "__fish_use_subcommand" -a "sft" -d "Fine-tuning with LlamaFactory"

# Global options
complete -c kt -l help -d "Show help message"
complete -c kt -l version -d "Show version"

# Run command options
complete -c kt -f -n "__fish_seen_subcommand_from run" -l host -d "Server host"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l port -d "Server port"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l gpu-experts -d "Number of GPU experts"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l cpu-threads -d "Number of CPU threads"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l tensor-parallel-size -d "Tensor parallel size"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l kt-method -d "KT method"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l attention-backend -d "Attention backend"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l max-total-tokens -d "Maximum total tokens"
complete -c kt -f -n "__fish_seen_subcommand_from run" -l dry-run -d "Show command without executing"

# Chat command options
complete -c kt -f -n "__fish_seen_subcommand_from chat" -l host -d "Server host"
complete -c kt -f -n "__fish_seen_subcommand_from chat" -l port -d "Server port"
complete -c kt -f -n "__fish_seen_subcommand_from chat" -l model -d "Model name"
complete -c kt -f -n "__fish_seen_subcommand_from chat" -l temperature -d "Sampling temperature"
complete -c kt -f -n "__fish_seen_subcommand_from chat" -l max-tokens -d "Maximum tokens"
complete -c kt -f -n "__fish_seen_subcommand_from chat" -l system -d "System prompt"
complete -c kt -f -n "__fish_seen_subcommand_from chat" -l save-history -d "Save conversation history"
complete -c kt -f -n "__fish_seen_subcommand_from chat" -l no-save-history -d "Do not save history"
complete -c kt -f -n "__fish_seen_subcommand_from chat" -l history-file -d "History file path"
complete -c kt -f -n "__fish_seen_subcommand_from chat" -l stream -d "Enable streaming output"
complete -c kt -f -n "__fish_seen_subcommand_from chat" -l no-stream -d "Disable streaming output"

# Quant command options
complete -c kt -f -n "__fish_seen_subcommand_from quant" -l method -d "Quantization method"
complete -c kt -f -n "__fish_seen_subcommand_from quant" -l output -d "Output directory"

# Bench command options
complete -c kt -f -n "__fish_seen_subcommand_from bench microbench" -l model -d "Model name or path"
complete -c kt -f -n "__fish_seen_subcommand_from bench microbench" -l config -d "Config file path"

# Doctor command options
complete -c kt -f -n "__fish_seen_subcommand_from doctor" -l verbose -d "Verbose output"

# Model subcommands
complete -c kt -f -n "__fish_seen_subcommand_from model; and not __fish_seen_subcommand_from download list path-list path-add path-remove search" -a "download" -d "Download a model from HuggingFace"
complete -c kt -f -n "__fish_seen_subcommand_from model; and not __fish_seen_subcommand_from download list path-list path-add path-remove search" -a "list" -d "List available models"
complete -c kt -f -n "__fish_seen_subcommand_from model; and not __fish_seen_subcommand_from download list path-list path-add path-remove search" -a "path-list" -d "List all model storage paths"
complete -c kt -f -n "__fish_seen_subcommand_from model; and not __fish_seen_subcommand_from download list path-list path-add path-remove search" -a "path-add" -d "Add a new model storage path"
complete -c kt -f -n "__fish_seen_subcommand_from model; and not __fish_seen_subcommand_from download list path-list path-add path-remove search" -a "path-remove" -d "Remove a model storage path"
complete -c kt -f -n "__fish_seen_subcommand_from model; and not __fish_seen_subcommand_from download list path-list path-add path-remove search" -a "search" -d "Search for models in the registry"

# Config subcommands
complete -c kt -f -n "__fish_seen_subcommand_from config; and not __fish_seen_subcommand_from show get set reset path init" -a "show" -d "Show all configuration"
complete -c kt -f -n "__fish_seen_subcommand_from config; and not __fish_seen_subcommand_from show get set reset path init" -a "get" -d "Get configuration value"
complete -c kt -f -n "__fish_seen_subcommand_from config; and not __fish_seen_subcommand_from show get set reset path init" -a "set" -d "Set configuration value"
complete -c kt -f -n "__fish_seen_subcommand_from config; and not __fish_seen_subcommand_from show get set reset path init" -a "reset" -d "Reset to defaults"
complete -c kt -f -n "__fish_seen_subcommand_from config; and not __fish_seen_subcommand_from show get set reset path init" -a "path" -d "Show configuration file path"
complete -c kt -f -n "__fish_seen_subcommand_from config; and not __fish_seen_subcommand_from show get set reset path init" -a "init" -d "Re-run first-time setup wizard"

# SFT subcommands
complete -c kt -f -n "__fish_seen_subcommand_from sft; and not __fish_seen_subcommand_from train chat export" -a "train" -d "Train model"
complete -c kt -f -n "__fish_seen_subcommand_from sft; and not __fish_seen_subcommand_from train chat export" -a "chat" -d "Chat with model"
complete -c kt -f -n "__fish_seen_subcommand_from sft; and not __fish_seen_subcommand_from train chat export" -a "export" -d "Export model"
