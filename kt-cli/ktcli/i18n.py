"""
Internationalization (i18n) module for kt-cli.

Supports English and Chinese languages, with automatic detection based on
system locale or KT_LANG environment variable.
"""

import os
from typing import Any

# Message definitions for all supported languages
MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        # General
        "welcome": "Welcome to KTransformers!",
        "goodbye": "Goodbye!",
        "error": "Error",
        "warning": "Warning",
        "success": "Success",
        "info": "Info",
        "yes": "Yes",
        "no": "No",
        "cancel": "Cancel",
        "confirm": "Confirm",
        "done": "Done",
        "failed": "Failed",
        "skip": "Skip",
        "back": "Back",
        "next": "Next",
        "retry": "Retry",
        "abort": "Abort",

        # Version command
        "version_info": "KTransformers CLI",
        "version_python": "Python",
        "version_platform": "Platform",
        "version_cuda": "CUDA",
        "version_cuda_not_found": "Not found",
        "version_kt_kernel": "kt-kernel",
        "version_ktransformers": "ktransformers",
        "version_sglang": "sglang",
        "version_llamafactory": "llamafactory",
        "version_not_installed": "Not installed",

        # Install command
        "install_detecting_env": "Detecting environment managers...",
        "install_found": "Found {name} (version {version})",
        "install_not_found": "Not found: {name}",
        "install_checking_env": "Checking existing environments...",
        "install_env_exists": "Found existing 'kt' environment",
        "install_env_not_exists": "No 'kt' environment found",
        "install_no_env_manager": "No virtual environment manager detected",
        "install_select_method": "Please select installation method:",
        "install_method_conda": "Create new conda environment 'kt' (Recommended)",
        "install_method_venv": "Create new venv environment",
        "install_method_uv": "Create new uv environment (Fast)",
        "install_method_docker": "Use Docker container",
        "install_method_system": "Install to system Python (Not recommended)",
        "install_select_mode": "Please select installation mode:",
        "install_mode_inference": "Inference - Install kt-kernel + SGLang",
        "install_mode_sft": "Training - Install kt-sft + LlamaFactory",
        "install_mode_full": "Full - Install all components",
        "install_creating_env": "Creating {type} environment '{name}'...",
        "install_env_created": "Environment created successfully",
        "install_installing_deps": "Installing dependencies...",
        "install_checking_deps": "Checking dependency versions...",
        "install_dep_ok": "OK",
        "install_dep_outdated": "Needs update",
        "install_dep_missing": "Missing",
        "install_deps_outdated": "Found {count} package(s) that need updating. Continue?",
        "install_updating": "Updating packages...",
        "install_complete": "Installation complete!",
        "install_activate_hint": "Activate environment: {command}",
        "install_start_hint": "Get started: kt run --help",
        "install_docker_pulling": "Pulling Docker image...",
        "install_docker_complete": "Docker image ready!",
        "install_docker_run_hint": "Run with: docker run --gpus all -p 30000:30000 {image} kt run {model}",
        "install_in_venv": "Running in virtual environment: {name}",
        "install_continue_without_venv": "Continue installing to system Python?",
        "install_already_installed": "All dependencies are already installed!",
        "install_confirm": "Install {count} package(s)?",

        # Config command
        "config_show_title": "Current Configuration",
        "config_set_success": "Configuration updated: {key} = {value}",
        "config_get_value": "{key} = {value}",
        "config_get_not_found": "Configuration key '{key}' not found",
        "config_reset_confirm": "This will reset all configurations to default. Continue?",
        "config_reset_success": "Configuration reset to default",
        "config_file_location": "Configuration file: {path}",

        # Doctor command
        "doctor_title": "KTransformers Environment Diagnostics",
        "doctor_checking": "Running diagnostics...",
        "doctor_check_python": "Python version",
        "doctor_check_cuda": "CUDA availability",
        "doctor_check_gpu": "GPU detection",
        "doctor_check_cpu": "CPU",
        "doctor_check_cpu_isa": "CPU Instructions",
        "doctor_check_numa": "NUMA Topology",
        "doctor_check_memory": "System memory",
        "doctor_check_disk": "Disk space",
        "doctor_check_packages": "Required packages",
        "doctor_check_env": "Environment variables",
        "doctor_status_ok": "OK",
        "doctor_status_warning": "Warning",
        "doctor_status_error": "Error",
        "doctor_gpu_found": "Found {count} GPU(s): {names}",
        "doctor_gpu_not_found": "No GPU detected",
        "doctor_cpu_info": "{name} ({cores} cores / {threads} threads)",
        "doctor_cpu_isa_info": "{isa_list}",
        "doctor_cpu_isa_missing": "Missing recommended: {missing}",
        "doctor_numa_info": "{nodes} node(s)",
        "doctor_numa_detail": "{node}: CPUs {cpus}",
        "doctor_memory_info": "{available} available / {total} total",
        "doctor_memory_freq": "{available} available / {total} total ({freq}MHz {type})",
        "doctor_disk_info": "{available} available at {path}",
        "doctor_all_ok": "All checks passed! Your environment is ready.",
        "doctor_has_issues": "Some issues were found. Please review the warnings/errors above.",

        # Run command
        "run_detecting_hardware": "Detecting hardware configuration...",
        "run_gpu_info": "GPU: {name} ({vram}GB VRAM)",
        "run_cpu_info": "CPU: {name} ({cores} cores, {numa} NUMA nodes)",
        "run_ram_info": "RAM: {total}GB",
        "run_checking_model": "Checking model status...",
        "run_model_path": "Model path: {path}",
        "run_weights_not_found": "Quantized weights not found",
        "run_quant_prompt": "Quantize model now? (This may take a while)",
        "run_quantizing": "Quantizing model...",
        "run_starting_server": "Starting server...",
        "run_server_mode": "Mode: SGLang + kt-kernel",
        "run_server_port": "Port: {port}",
        "run_gpu_experts": "GPU experts: {count}/layer",
        "run_cpu_threads": "CPU threads: {count}",
        "run_server_started": "Server started!",
        "run_api_url": "API URL: http://{host}:{port}",
        "run_docs_url": "Docs URL: http://{host}:{port}/docs",
        "run_stop_hint": "Press Ctrl+C to stop the server",
        "run_model_not_found": "Model '{name}' not found. Run 'kt download' first.",
        "run_multiple_matches": "Multiple models found. Please select:",
        "run_select_model": "Select model",

        # Download command
        "download_list_title": "Available Models",
        "download_searching": "Searching for model '{name}'...",
        "download_found": "Found: {name}",
        "download_multiple_found": "Multiple matches found:",
        "download_select": "Select model to download:",
        "download_destination": "Destination: {path}",
        "download_starting": "Starting download...",
        "download_progress": "Downloading {name}...",
        "download_complete": "Download complete!",
        "download_already_exists": "Model already exists at {path}",
        "download_overwrite_prompt": "Overwrite existing files?",

        # Quant command
        "quant_input_path": "Input path: {path}",
        "quant_output_path": "Output path: {path}",
        "quant_method": "Quantization method: {method}",
        "quant_starting": "Starting quantization...",
        "quant_progress": "Quantizing...",
        "quant_complete": "Quantization complete!",
        "quant_input_not_found": "Input model not found at {path}",

        # SFT command
        "sft_mode_train": "Training mode",
        "sft_mode_chat": "Chat mode",
        "sft_mode_export": "Export mode",
        "sft_config_path": "Config file: {path}",
        "sft_starting": "Starting {mode}...",
        "sft_complete": "{mode} complete!",
        "sft_config_not_found": "Config file not found: {path}",

        # Bench command
        "bench_starting": "Starting benchmark...",
        "bench_type": "Benchmark type: {type}",
        "bench_complete": "Benchmark complete!",
        "bench_results_title": "Benchmark Results",

        # Common prompts
        "prompt_continue": "Continue?",
        "prompt_select": "Please select:",
        "prompt_enter_value": "Enter value:",
        "prompt_confirm_action": "Confirm this action?",

        # First-run setup - Model path selection
        "setup_model_path_title": "Model Storage Location",
        "setup_model_path_desc": "LLM models are large (50-200GB+). Please select a storage location with sufficient space:",
        "setup_scanning_disks": "Scanning available storage locations...",
        "setup_disk_option": "{path} ({available} available / {total} total)",
        "setup_disk_option_recommended": "{path} ({available} available / {total} total) [Recommended]",
        "setup_custom_path": "Enter custom path",
        "setup_enter_custom_path": "Enter the path for model storage",
        "setup_path_not_exist": "Path does not exist. Create it?",
        "setup_path_no_write": "No write permission for this path. Please choose another.",
        "setup_path_low_space": "Warning: Less than 100GB available. Large models may not fit.",
        "setup_model_path_set": "Model storage path set to: {path}",
        "setup_no_large_disk": "No large storage locations found. Using default path.",
        "setup_scanning_models": "Scanning for existing models...",
        "setup_found_models": "Found {count} model(s):",
        "setup_model_info": "{name} ({size}, {type})",
        "setup_no_models_found": "No existing models found in this location.",
        "setup_location_has_models": "{count} model(s) found",
    },
    "zh": {
        # General
        "welcome": "欢迎使用 KTransformers！",
        "goodbye": "再见！",
        "error": "错误",
        "warning": "警告",
        "success": "成功",
        "info": "信息",
        "yes": "是",
        "no": "否",
        "cancel": "取消",
        "confirm": "确认",
        "done": "完成",
        "failed": "失败",
        "skip": "跳过",
        "back": "返回",
        "next": "下一步",
        "retry": "重试",
        "abort": "中止",

        # Version command
        "version_info": "KTransformers CLI",
        "version_python": "Python",
        "version_platform": "平台",
        "version_cuda": "CUDA",
        "version_cuda_not_found": "未找到",
        "version_kt_kernel": "kt-kernel",
        "version_ktransformers": "ktransformers",
        "version_sglang": "sglang",
        "version_llamafactory": "llamafactory",
        "version_not_installed": "未安装",

        # Install command
        "install_detecting_env": "检测环境管理工具...",
        "install_found": "发现 {name} (版本 {version})",
        "install_not_found": "未找到: {name}",
        "install_checking_env": "检查现有环境...",
        "install_env_exists": "发现现有 'kt' 环境",
        "install_env_not_exists": "未发现 'kt' 环境",
        "install_no_env_manager": "未检测到虚拟环境管理工具",
        "install_select_method": "请选择安装方式:",
        "install_method_conda": "创建新的 conda 环境 'kt' (推荐)",
        "install_method_venv": "创建新的 venv 环境",
        "install_method_uv": "创建新的 uv 环境 (快速)",
        "install_method_docker": "使用 Docker 容器",
        "install_method_system": "安装到系统 Python (不推荐)",
        "install_select_mode": "请选择安装模式:",
        "install_mode_inference": "推理模式 - 安装 kt-kernel + SGLang",
        "install_mode_sft": "训练模式 - 安装 kt-sft + LlamaFactory",
        "install_mode_full": "完整安装 - 安装所有组件",
        "install_creating_env": "正在创建 {type} 环境 '{name}'...",
        "install_env_created": "环境创建成功",
        "install_installing_deps": "正在安装依赖...",
        "install_checking_deps": "检查依赖版本...",
        "install_dep_ok": "正常",
        "install_dep_outdated": "需更新",
        "install_dep_missing": "缺失",
        "install_deps_outdated": "发现 {count} 个包需要更新，是否继续？",
        "install_updating": "正在更新包...",
        "install_complete": "安装完成！",
        "install_activate_hint": "激活环境: {command}",
        "install_start_hint": "开始使用: kt run --help",
        "install_docker_pulling": "正在拉取 Docker 镜像...",
        "install_docker_complete": "Docker 镜像已就绪！",
        "install_docker_run_hint": "运行: docker run --gpus all -p 30000:30000 {image} kt run {model}",
        "install_in_venv": "当前在虚拟环境中: {name}",
        "install_continue_without_venv": "继续安装到系统 Python？",
        "install_already_installed": "所有依赖已安装！",
        "install_confirm": "安装 {count} 个包？",

        # Config command
        "config_show_title": "当前配置",
        "config_set_success": "配置已更新: {key} = {value}",
        "config_get_value": "{key} = {value}",
        "config_get_not_found": "未找到配置项 '{key}'",
        "config_reset_confirm": "这将重置所有配置为默认值。是否继续？",
        "config_reset_success": "配置已重置为默认值",
        "config_file_location": "配置文件: {path}",

        # Doctor command
        "doctor_title": "KTransformers 环境诊断",
        "doctor_checking": "正在运行诊断...",
        "doctor_check_python": "Python 版本",
        "doctor_check_cuda": "CUDA 可用性",
        "doctor_check_gpu": "GPU 检测",
        "doctor_check_cpu": "CPU",
        "doctor_check_cpu_isa": "CPU 指令集",
        "doctor_check_numa": "NUMA 拓扑",
        "doctor_check_memory": "系统内存",
        "doctor_check_disk": "磁盘空间",
        "doctor_check_packages": "必需的包",
        "doctor_check_env": "环境变量",
        "doctor_status_ok": "正常",
        "doctor_status_warning": "警告",
        "doctor_status_error": "错误",
        "doctor_gpu_found": "发现 {count} 个 GPU: {names}",
        "doctor_gpu_not_found": "未检测到 GPU",
        "doctor_cpu_info": "{name} ({cores} 核心 / {threads} 线程)",
        "doctor_cpu_isa_info": "{isa_list}",
        "doctor_cpu_isa_missing": "缺少推荐指令集: {missing}",
        "doctor_numa_info": "{nodes} 个节点",
        "doctor_numa_detail": "{node}: CPU {cpus}",
        "doctor_memory_info": "{available} 可用 / {total} 总计",
        "doctor_memory_freq": "{available} 可用 / {total} 总计 ({freq}MHz {type})",
        "doctor_disk_info": "{path} 有 {available} 可用空间",
        "doctor_all_ok": "所有检查通过！您的环境已就绪。",
        "doctor_has_issues": "发现一些问题，请查看上方的警告/错误信息。",

        # Run command
        "run_detecting_hardware": "检测硬件配置...",
        "run_gpu_info": "GPU: {name} ({vram}GB 显存)",
        "run_cpu_info": "CPU: {name} ({cores} 核心, {numa} NUMA 节点)",
        "run_ram_info": "内存: {total}GB",
        "run_checking_model": "检查模型状态...",
        "run_model_path": "模型路径: {path}",
        "run_weights_not_found": "未找到量化权重",
        "run_quant_prompt": "是否现在量化模型？(这可能需要一些时间)",
        "run_quantizing": "正在量化模型...",
        "run_starting_server": "正在启动服务器...",
        "run_server_mode": "模式: SGLang + kt-kernel",
        "run_server_port": "端口: {port}",
        "run_gpu_experts": "GPU 专家: {count}/层",
        "run_cpu_threads": "CPU 线程: {count}",
        "run_server_started": "服务器已启动！",
        "run_api_url": "API 地址: http://{host}:{port}",
        "run_docs_url": "文档地址: http://{host}:{port}/docs",
        "run_stop_hint": "按 Ctrl+C 停止服务器",
        "run_model_not_found": "未找到模型 '{name}'。请先运行 'kt download'。",
        "run_multiple_matches": "找到多个匹配的模型，请选择:",
        "run_select_model": "选择模型",

        # Download command
        "download_list_title": "可用模型",
        "download_searching": "正在搜索模型 '{name}'...",
        "download_found": "找到: {name}",
        "download_multiple_found": "找到多个匹配:",
        "download_select": "选择要下载的模型:",
        "download_destination": "目标路径: {path}",
        "download_starting": "开始下载...",
        "download_progress": "正在下载 {name}...",
        "download_complete": "下载完成！",
        "download_already_exists": "模型已存在于 {path}",
        "download_overwrite_prompt": "是否覆盖现有文件？",

        # Quant command
        "quant_input_path": "输入路径: {path}",
        "quant_output_path": "输出路径: {path}",
        "quant_method": "量化方法: {method}",
        "quant_starting": "开始量化...",
        "quant_progress": "正在量化...",
        "quant_complete": "量化完成！",
        "quant_input_not_found": "未找到输入模型: {path}",

        # SFT command
        "sft_mode_train": "训练模式",
        "sft_mode_chat": "聊天模式",
        "sft_mode_export": "导出模式",
        "sft_config_path": "配置文件: {path}",
        "sft_starting": "正在启动 {mode}...",
        "sft_complete": "{mode} 完成！",
        "sft_config_not_found": "未找到配置文件: {path}",

        # Bench command
        "bench_starting": "开始基准测试...",
        "bench_type": "测试类型: {type}",
        "bench_complete": "基准测试完成！",
        "bench_results_title": "基准测试结果",

        # Common prompts
        "prompt_continue": "是否继续？",
        "prompt_select": "请选择:",
        "prompt_enter_value": "请输入:",
        "prompt_confirm_action": "确认此操作？",

        # First-run setup - Model path selection
        "setup_model_path_title": "模型存储位置",
        "setup_model_path_desc": "大语言模型体积较大（50-200GB+）。请选择一个有足够空间的存储位置：",
        "setup_scanning_disks": "正在扫描可用存储位置...",
        "setup_disk_option": "{path} (可用 {available} / 总共 {total})",
        "setup_disk_option_recommended": "{path} (可用 {available} / 总共 {total}) [推荐]",
        "setup_custom_path": "输入自定义路径",
        "setup_enter_custom_path": "请输入模型存储路径",
        "setup_path_not_exist": "路径不存在，是否创建？",
        "setup_path_no_write": "没有该路径的写入权限，请选择其他路径。",
        "setup_path_low_space": "警告：可用空间不足 100GB，可能无法存储大型模型。",
        "setup_model_path_set": "模型存储路径已设置为: {path}",
        "setup_no_large_disk": "未发现大容量存储位置，使用默认路径。",
        "setup_scanning_models": "正在扫描已有模型...",
        "setup_found_models": "发现 {count} 个模型:",
        "setup_model_info": "{name} ({size}, {type})",
        "setup_no_models_found": "该位置未发现已有模型。",
        "setup_location_has_models": "发现 {count} 个模型",
    },
}


def get_lang() -> str:
    """
    Detect the current language setting.

    Priority:
    1. KT_LANG environment variable
    2. Config file general.language setting
    3. LANG environment variable (if config is "auto")
    4. Default to English

    Returns:
        Language code: "zh" for Chinese, "en" for English
    """
    # 1. Check KT_LANG environment variable (highest priority)
    kt_lang = os.environ.get("KT_LANG", "").lower()
    if kt_lang:
        return "zh" if kt_lang.startswith("zh") else "en"

    # 2. Check config file setting
    # Import here to avoid circular imports
    from ktcli.config.settings import get_settings
    try:
        settings = get_settings()
        config_lang = settings.get("general.language", "auto")
        if config_lang and config_lang != "auto":
            return "zh" if config_lang.lower().startswith("zh") else "en"
    except Exception:
        # If settings fail to load, continue with system detection
        pass

    # 3. Check system LANG environment variable
    system_lang = os.environ.get("LANG", "").lower()
    if system_lang.startswith("zh"):
        return "zh"

    return "en"


def t(msg_key: str, **kwargs: Any) -> str:
    """
    Translate a message key to the current language.

    Args:
        msg_key: Message key to translate
        **kwargs: Format arguments for the message

    Returns:
        Translated and formatted message string

    Example:
        >>> t("welcome")
        "Welcome to KTransformers!"  # or "欢迎使用 KTransformers！" in Chinese

        >>> t("install_found", name="conda", version="24.1.0")
        "Found conda (version 24.1.0)"
    """
    lang = get_lang()
    messages = MESSAGES.get(lang, MESSAGES["en"])
    message = messages.get(msg_key, MESSAGES["en"].get(msg_key, msg_key))

    if kwargs:
        try:
            return message.format(**kwargs)
        except KeyError:
            return message
    return message


def set_lang(lang: str) -> None:
    """
    Set the language for the current session.

    Args:
        lang: Language code ("en" or "zh")
    """
    os.environ["KT_LANG"] = lang
