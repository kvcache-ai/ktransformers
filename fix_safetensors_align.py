#!/usr/bin/env python3
"""
修复 safetensors 文件使 data_start 512 对齐，满足 O_DIRECT 要求。

safetensors 格式：8字节header_len + JSON header + 数据部分
data_start = 8 + header_len
O_DIRECT 要求 offset 512 对齐，所以 data_start % 512 == 0

修复方法：在 JSON header 末尾填充空格，使 header_len 增加 padding 字节
JSON 规范允许顶层对象后尾随空白，safetensors 库的 json.loads 能正确处理

用法：python3 fix_safetensors_align.py <safetensors_dir> [--dry-run]
"""
import struct
import os
import sys
import json
import tempfile
import shutil

def fix_one_file(fpath: str, dry_run: bool = False) -> bool:
    """修复单个 safetensors 文件。返回 True if 已修复或已对齐。"""
    file_size = os.path.getsize(fpath)
    with open(fpath, 'rb') as fp:
        header_len = struct.unpack('<Q', fp.read(8))[0]
        header_json = fp.read(header_len).decode('utf-8')
        data_start = 8 + header_len
        align = data_start % 512
        if align == 0:
            print(f"[SKIP] {os.path.basename(fpath)}: already aligned (data_start={data_start})")
            return True
        padding_needed = (512 - align) % 512
        # 验证 JSON 可解析
        try:
            header_obj = json.loads(header_json)
        except json.JSONDecodeError as e:
            print(f"[ERROR] {os.path.basename(fpath)}: invalid JSON header: {e}")
            return False
        # 添加 padding 空格到 JSON header 末尾
        new_header_json = header_json + (' ' * padding_needed)
        new_header_len = header_len + padding_needed
        new_data_start = 8 + new_header_len
        assert new_data_start % 512 == 0, f"alignment check failed: {new_data_start % 512}"
        print(f"[FIX] {os.path.basename(fpath)}: header_len {header_len}->{new_header_len}, "
              f"data_start {data_start}->{new_data_start}, padding={padding_needed}")
        if dry_run:
            return True
        # 创建临时文件（同目录，确保同文件系统，方便原子替换）
        tmp_dir = os.path.dirname(fpath)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.tmp_align', dir=tmp_dir)
        try:
            with os.fdopen(tmp_fd, 'wb') as tmp:
                # 写入新的 header_len
                tmp.write(struct.pack('<Q', new_header_len))
                # 写入新的 JSON header（带 padding 空格）
                tmp.write(new_header_json.encode('utf-8'))
                # 拷贝数据部分（从原始文件的 data_start 开始到文件末尾）
                with open(fpath, 'rb') as src:
                    src.seek(data_start)
                    # 使用 sendfile 高效拷贝（零拷贝）
                    remaining = file_size - data_start
                    while remaining > 0:
                        chunk = min(remaining, 64 * 1024 * 1024)  # 64MB chunks
                        n = sendfile_chunk(tmp.fileno(), src.fileno(), chunk)
                        if n == 0:
                            raise IOError("sendfile returned 0 bytes")
                        remaining -= n
            # 原子替换
            os.replace(tmp_path, fpath)
            print(f"[OK] {os.path.basename(fpath)}: fixed and replaced")
            return True
        except Exception as e:
            print(f"[ERROR] {os.path.basename(fpath)}: {e}")
            try:
                os.unlink(tmp_path)
            except:
                pass
            return False

def sendfile_chunk(dst_fd: int, src_fd: int, count: int) -> int:
    """使用 os.sendfile 拷贝数据，回退到 read/write。"""
    try:
        import sendfile  # not standard
        return sendfile.sendfile(dst_fd, src_fd, 0, count)
    except ImportError:
        pass
    # 使用 os.splice 或 os.copy_file_range（Linux 专用）
    try:
        # os.copy_file_range 在 Python 3.5+ 可用
        n = os.copy_file_range(src_fd, dst_fd, count)
        return n
    except (AttributeError, OSError):
        pass
    # 回退到 read/write
    data = os.read(src_fd, count)
    os.write(dst_fd, data)
    return len(data)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <safetensors_dir> [--dry-run]")
        sys.exit(1)
    dir_path = sys.argv[1]
    dry_run = '--dry-run' in sys.argv
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a directory")
        sys.exit(1)
    files = sorted([f for f in os.listdir(dir_path) if f.endswith('.safetensors')])
    if not files:
        print(f"Error: no .safetensors files in {dir_path}")
        sys.exit(1)
    print(f"Found {len(files)} safetensors files in {dir_path}")
    if dry_run:
        print("[DRY RUN] no files will be modified")
    print()
    success = 0
    for f in files:
        fpath = os.path.join(dir_path, f)
        if fix_one_file(fpath, dry_run):
            success += 1
    print(f"\nDone: {success}/{len(files)} files aligned")

if __name__ == '__main__':
    main()
