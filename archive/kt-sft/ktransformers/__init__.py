#!/usr/bin/env python
# coding=utf-8
'''
Description  :
Author       : kkk1nak0
Date         : 2024-08-15 07:34:46
Version      : 1.0.0
LastEditors  : chenxl
LastEditTime : 2025-02-15 03:53:02
'''
import sys
import os

# Import version from shared version.py at project root
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root_dir)
try:
    from version import __version__
finally:
    sys.path.pop(0)
