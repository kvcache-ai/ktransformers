import os
import re
import socket
import random
from copy import deepcopy
from typing import Sequence

import yaml


def get_sh_body():
    sh = r'''#!/bin/bash
set -ex

# export area
{export_area}

# source area
{source_area}

# global vars
LOG_DIR="../logs"
{log_file_name}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p ${LOG_DIR}
LOG_FILE="${LOG_DIR}/${LOG_NAME}_${TIMESTAMP}.log"

# torchrun and model area
torchrun \
  --master-port {master_port} \
{torchrun_area}
{model_area}
2>&1 | tee "${LOG_FILE}"'''
    return sh


def add_static_pattern_seq(sh: str, data: Sequence, pattern_name: str, replace_line: str):
    lines = [replace_line.format(val) for val in data]
    replace_context = "\n".join(lines)
    return sh.replace(pattern_name, replace_context)


def add_static_pattern_dict(sh: str, data: dict, pattern_name: str, replace_line: str):
    lines = [replace_line.format(key, val) for key, val in data.items()]
    replace_context = "\n".join(lines)
    return sh.replace(pattern_name, replace_context)


def add_export(sh, data: dict):
    return add_static_pattern_dict(sh, data, "{export_area}", "export {0}={1}")


def add_source(sh, data: Sequence):
    return add_static_pattern_seq(sh, data, "{source_area}", "source {0}")


def add_torchrun(sh, data):
    long_replace = add_static_pattern_dict("long", data["long"], "long", "  --{0} {1} \\")
    short_replace = add_static_pattern_dict("short", data["short"], "short", "  -{0} {1} \\")
    return sh.replace("{torchrun_area}", long_replace + "\n" + short_replace)


def add_model(sh, data: dict):
    return add_static_pattern_dict(sh, data, "{model_area}", "  --{0} {1} \\")


def get_valid_file_name_sequence(s: str) -> str:
    return "".join(char for char in s if char.isalnum())


def get_valid_file_name_lines(hyper):
    keys_required = ("cpu_infer", )
    if not all(key in hyper for key in keys_required):
        raise ValueError(f"{', '.join(keys_required)} should be in hyperparams of generator.py to generate file name.")
    ret = [get_valid_file_name_sequence(f"{key}{hyper[key]}") for key in keys_required]
    return "test_" + "_".join(ret)


def add_log_file(sh, hyper):
    hyperparams_lines = get_valid_file_name_lines(hyper)
    return sh.replace("{log_file_name}", "LOG_NAME=\"" + hyperparams_lines + "\"")


def is_available_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except socket.error as e:
            if e.errno == 98:
                return False
            raise e

def add_available_port(sh, start=1024, end=65535):
    while True:
        port = random.randint(start, end)
        if is_available_port(port):
            return sh.replace("{master_port}", f"{port}")


def get_all_hyper_keys(hypers):
    keys = set()
    for hyper in hypers:
        for key, val in hyper.items():
            keys.add(key)
    return keys


def replace_hyper(sh: str, hyper, all_hyper_keys):
    lines = []
    hyper_not_used = set(hyper.keys())
    for line in sh.split("\n"):
        match = re.search("\$([a-zA-Z0-9_]+)", line)
        if not match:
            lines.append(line)
            continue

        key = match.group(1)
        if key in hyper:
            hyper_not_used.remove(key)
            lines.append(line.replace(f"${key}", str(hyper[key])))
        elif key not in all_hyper_keys:
            print(f"[WARNING] `{key}` not in hyperparams, will skip generating the line.")

    if hyper_not_used:
        print(f"[WARNING] The following hyperparams are not used: {','.join(hyper_not_used)}")
    return "\n".join(lines)


def load_yaml_file(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError("Failed to find config file on {}".format(config_path))

    with open(config_path) as f:
        return yaml.safe_load(f)

def save_sh_file(sh, fname):
    with open(fname, "w") as f:
        f.write(sh)
    os.chmod(fname, 0o550)
    print("Generate file: ", fname)


if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.dirname(__file__)) + "/"
    y = load_yaml_file(root_dir + "test_generator.yaml")
    hypers = y["hyperparams"]
    all_hyper_keys = get_all_hyper_keys(hypers)
    test_cnt = len(hypers)

    sh = get_sh_body()
    sh = add_export(sh, y["export_area"])
    sh = add_source(sh, y["source_area"])
    sh = add_torchrun(sh, y["torchrun_area"])
    sh = add_model(sh, y["model_area"])

    for hyper in hypers:
        sh_ = deepcopy(sh)
        sh_ = add_available_port(sh_)
        sh_ = add_log_file(sh_, hyper)
        sh_ = replace_hyper(sh_, hyper, all_hyper_keys)
        fname = root_dir + get_valid_file_name_lines(hyper) + ".sh"
        save_sh_file(sh_, fname)
