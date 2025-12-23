#!/bin/bash
# Bash completion for kt command
# This is a static completion script that doesn't require Python startup

_kt_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main commands
    local commands="version install update run download quant bench microbench doctor model config sft"

    # Global options
    local global_opts="--help --version"

    # Handle subcommands
    case "${COMP_CWORD}" in
        1)
            # First argument: suggest commands and global options
            COMPREPLY=( $(compgen -W "${commands} ${global_opts}" -- ${cur}) )
            return 0
            ;;
        *)
            # Handle specific command options
            case "${COMP_WORDS[1]}" in
                install)
                    local install_opts="--source --branch --yes --force --skip-torch --editable --from-source --cpu-instruct --enable-amx --disable-amx --build-type --deps-only --docker --verify --no-verify --help"
                    COMPREPLY=( $(compgen -W "${install_opts} inference sft full" -- ${cur}) )
                    ;;
                update)
                    local update_opts="--source --pypi --yes --help"
                    COMPREPLY=( $(compgen -W "${update_opts}" -- ${cur}) )
                    ;;
                run)
                    local run_opts="--model --config --host --port --gpu-experts --cpu-threads --help"
                    COMPREPLY=( $(compgen -W "${run_opts}" -- ${cur}) )
                    ;;
                download)
                    local download_opts="--output --resume --mirror --help"
                    COMPREPLY=( $(compgen -W "${download_opts}" -- ${cur}) )
                    ;;
                quant)
                    local quant_opts="--method --output --help"
                    COMPREPLY=( $(compgen -W "${quant_opts}" -- ${cur}) )
                    ;;
                bench|microbench)
                    local bench_opts="--model --config --help"
                    COMPREPLY=( $(compgen -W "${bench_opts}" -- ${cur}) )
                    ;;
                doctor)
                    local doctor_opts="--verbose --help"
                    COMPREPLY=( $(compgen -W "${doctor_opts}" -- ${cur}) )
                    ;;
                model)
                    local model_cmds="download list path-list path-add path-remove search"
                    local model_opts="--help"
                    COMPREPLY=( $(compgen -W "${model_cmds} ${model_opts}" -- ${cur}) )
                    ;;
                config)
                    local config_cmds="show get set reset path init model-path-list model-path-add model-path-remove"
                    local config_opts="--help"
                    COMPREPLY=( $(compgen -W "${config_cmds} ${config_opts}" -- ${cur}) )
                    ;;
                sft)
                    local sft_cmds="train chat export"
                    local sft_opts="--help"
                    COMPREPLY=( $(compgen -W "${sft_cmds} ${sft_opts}" -- ${cur}) )
                    ;;
                version)
                    COMPREPLY=( $(compgen -W "--help" -- ${cur}) )
                    ;;
                *)
                    COMPREPLY=()
                    ;;
            esac
            ;;
    esac
}

complete -F _kt_completion kt
