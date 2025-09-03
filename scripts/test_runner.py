import os
import select
import subprocess
import time
from functools import partial
from types import SimpleNamespace
from typing import Sequence, Tuple, Dict, Union

import psutil
import yaml


def load_yaml_file(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError('[Runner] Failed to find config file on {}'.format(config_path))

    with open(config_path) as f:
        return yaml.safe_load(f)


def store_output_stat(line: str, store: dict[str, list[str]]) -> bool:
    line_head = line[:30]
    for key in store.keys():
        if key in line_head:
            store[key].append(line)
            return True
    return False


def get_file_line_cnt(line):
    if os.path.isfile(line):
        with open(line, 'r') as f:
            non_empty_lines = sum(1 for line in f if line.strip())
    else:
        non_empty_lines = 1
        print(f'[Runner] Use default line count: 1')
    return non_empty_lines


def print_script_running(proc, script, args, script_id) -> tuple[bool, str | dict] | None:
    is_print_lines = args.is_print_lines
    print_interval = args.print_interval
    test_input = script.get('test_input', 'Hello, world.')
    timeout = script.get('timeout', 20000)  # sec

    line_cnt = get_file_line_cnt(test_input)
    max_stat_count = line_cnt * len(args.stat_keys)
    print(f'[Runner] The script will exit after the stat info in the {max_stat_count} line is collected.')
    stat_count = 0
    success_collect = False
    test_output = {key: [] for key in args.stat_keys}

    failure_start_time = 0
    failure_wait_time = 3
    failure_ret_info = ''

    buffer = ''
    prompt_detected = False
    last_status_time = start_time = time.time()

    os.set_blocking(proc.stdout.fileno(), False)

    read_chunk = partial(os.read, proc.stdout.fileno(), 4096)

    def add_os_linesep(text):
        if not text.endswith(os.linesep):
            text += os.linesep
        return text

    def work_on_detected():
        nonlocal prompt_detected
        print(f'[Runner] Chat detected')
        proc.stdin.write(add_os_linesep(test_input))
        proc.stdin.flush()
        prompt_detected = True

    def update_buffer():
        try:
            nonlocal buffer
            while True:
                data = read_chunk().decode('utf-8', errors='replace')
                if not data:
                    break
                buffer += data
        except BlockingIOError:
            pass  # Buffer is empty

    def process_lines(lines):
        nonlocal failure_start_time, failure_ret_info, stat_count, is_print_lines
        for line in lines:
            if is_print_lines:
                print(line)

            if 'Traceback' in line and failure_ret_info == '':
                failure_start_time = cur_time
                failure_ret_info = f'Detect traceback, exiting...'
                print(f'[Runner] Detect traceback')
                print(line)
                is_print_lines = True

            if 'NPU out of memory' in line:
                failure_ret_info = f'Detect NPU OOM, exiting...'

            if store_output_stat(line, test_output):
                stat_count += 1

            if stat_count >= max_stat_count:
                nonlocal success_collect
                success_collect = True

            if not prompt_detected and 'Chat:' in line:
                work_on_detected()

    def process_incomplete_line(line):
        if not prompt_detected and 'Chat:' in line:
            if is_print_lines:
                print(line)
            work_on_detected()
            line = ''
        return line

    while True:
        cur_time = time.time()
        cur_running_time = int(cur_time - start_time)
        if cur_time - last_status_time > print_interval:
            print(f'[Runner] Script still running (elapsed: {cur_running_time}s), task id: {script_id}')
            last_status_time = cur_time

        if cur_time - start_time > timeout:
            print(f'[Runner] Timeout, exiting...')
            return False, f'Script execution timeout after {cur_running_time}s'

        if success_collect:
            return True, test_output

        if failure_ret_info and cur_time - failure_start_time >= failure_wait_time:
            return False, failure_ret_info

        rlist, _, _ = select.select([proc.stdout], [], [], 0.2)

        if rlist:
            update_buffer()

            if buffer:
                lines = buffer.split('\n')

                process_lines(lines[:-1])

                buffer = process_incomplete_line(lines[-1])


def kill_process(p):
    try:
        if p.status() == psutil.STATUS_ZOMBIE:
            print(f'[Runner] Found zombie process {p.pid}, waiting parent to reap it')
            os.waitpid(p.pid, os.WNOHANG)
        else:
            p.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def kill_subprocess(args):
    pid = os.getpid()
    self = psutil.Process(pid)

    for _ in range(10):
        children = self.children(recursive=True)
        if not children:
            print('[Runner] Script exiting successfully.')
            time.sleep(2)
            return

        print(f'[Runner] Killing {len(children)} subprocesses...')

        for child in children:
            kill_process(child)

        time.sleep(args.sleep_time)

    raise RuntimeError('[Runner] Subprocess exited unsuccessfully!')


def script_runner(script: dict[str, str], args, script_id) -> tuple[bool, Union[str, dict | Sequence]]:
    max_retries = args.max_retries
    script_path = script['path']
    if not os.path.isabs(script_path):
        script_path = os.path.join(args.runner_path, script_path)
    retries = 0
    ret_failure = []

    while retries < max_retries:
        print(f'[Runner] Running: {script_path}' + ('' if retries == 0 else f'(attempt {retries + 1}/{max_retries})'))
        proc = subprocess.Popen(
            ['/bin/bash', '--noprofile', '--norc', script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=4096,
            text=True
        )

        stat, ret = print_script_running(proc, script, args, script_id)

        proc.stdin.close()
        proc.stdout.close()
        proc.wait()

        kill_subprocess(args)

        if stat:
            return True, ret

        if not isinstance(ret, str):
            raise AssertionError('ret must be a string when run failure.')
        ret_failure.append(ret)
        if 'Script execution timeout after' in ret:
            print('[Runner] Timeout.')
        elif 'Detect traceback, exiting...' in ret:
            print('[Runner] Get traceback.')
        elif 'Detect NPU OOM, exiting...' in ret:
            print('[Runner] Get NPU OOM.')
        else:
            print(f'[Runner] Get error: {ret}')

        if args.stop_on_failure:
            return False, ret_failure

        retries += 1
        if retries < max_retries:
            continue

        return False, ret_failure

    return False, ('[RunnerError] max_retries must greater than 0.',)


def print_centered_summary(text, char='=', width=60):
    if len(text) >= width - 10 - 2:
        print(f'\n{char * 5} {text} {char * 5}\n', end='')
    else:
        padding = (width - 2 - len(text)) // 2
        left_padding = char * padding
        right_padding = char * (width - len(text) - padding)
        print(f'\n{left_padding} {text} {right_padding}\n', end='')


def print_success_output(output):
    for script_name, vals in output.items():
        print_centered_summary(script_name, '-')
        if len(vals) == 0:
            return
        len_val = len(next(iter(vals.values())))
        for idx in range(len_val):
            for val in vals.values():
                print(val[idx], end='\n')
            print()


def print_failure_output(output):
    for script_name, vals in output.items():
        print_centered_summary(script_name, '-')
        if len(vals) == 0:
            return
        for val in vals:
            print(val, end='\n')
        print()


def make_titles(scripts):
    names_cnt = {}
    global_cnt = 0
    titles = []
    for item in scripts:
        name = item['name']
        names_cnt[name] = names_cnt.get(name, 0) + 1
        global_cnt += 1
        titles.append(f'{global_cnt} {name}' + ('' if names_cnt[name] == 1 else f' ({names_cnt[name]})'))
    return titles


if __name__ == '__main__':
    runner_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/'

    config = load_yaml_file(runner_path + 'test_runner.yaml')
    settings = config['settings']
    scripts = config['scripts']
    stat_keys = config['stat_keys']
    titles = make_titles(scripts)
    if len(scripts) == 0:
        print('[Runner] No scripts defined.')
        exit(1)

    args = SimpleNamespace()
    args.stat_keys = stat_keys
    args.max_retries = settings.get('max_retries', 3)
    args.sleep_time = settings.get('sleep_time', 30)  # second
    args.is_print_lines = settings.get('is_print_lines', True)
    args.print_interval = settings.get('print_interval', 30)  # second
    args.stop_on_failure = settings.get('stop_on_failure', False)
    args.runner_path = runner_path
    args.root_path = root_path

    success_output = {}
    failure_output = {}
    try:
        for i, script in enumerate(scripts):
            stat, ret = script_runner(script, args, i + 1)
            if stat:
                success_output[titles[i]] = ret
            else:
                failure_output[titles[i]] = ret
            if args.stop_on_failure and not stat:
                print(f'[Runner] Running {titles[i]} failed. Exit early now because stop_on_failure is set to True.')
                break
    except Exception as e:
        if '[Runner' in e.__str__():
            print(f'{e.__str__()} \n[Runner] Detect error, the collected information will be print.')
        else:
            raise e

    print_centered_summary('Summary Begin')
    print_success_output(success_output)
    print_failure_output(failure_output)
    success_failure_stat = []
    if len(success_output) > 0:
        success_failure_stat.append(f'{len(success_output)} scripts execute success')
    if len(failure_output) > 0:
        success_failure_stat.append(f'{len(failure_output)} scripts execute failure')
    print(', '.join(success_failure_stat) + '.')
    print_centered_summary('Summary End')
