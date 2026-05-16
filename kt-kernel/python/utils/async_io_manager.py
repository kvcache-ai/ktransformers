"""
Global AsyncExpertReader singleton manager for io_uring-based expert loading.
"""

import os

import kt_kernel_ext as ext

_global_async_reader = None
_global_async_readers = []


def get_global_async_reader():
    """
    Get or create the global AsyncExpertReader singleton.

    Returns:
        AsyncExpertReader: Global async reader instance
    """
    global _global_async_reader
    if _global_async_reader is None:
        queue_depth = int(os.environ.get("KT_IOURING_QUEUE_DEPTH", "1024"))
        _global_async_reader = ext.AsyncExpertReader(queue_depth=queue_depth)
    return _global_async_reader


def get_async_readers(count):
    """
    Get or create one AsyncExpertReader per TP/NUMA shard.

    Each reader owns an independent io_uring ring and I/O thread, so TP shards
    do not serialize on one shared ring owner.
    """
    global _global_async_readers
    count = max(1, int(count))
    queue_depth = int(os.environ.get("KT_IOURING_QUEUE_DEPTH", "1024"))
    while len(_global_async_readers) < count:
        _global_async_readers.append(ext.AsyncExpertReader(queue_depth=queue_depth))
    return _global_async_readers[:count]


def shutdown_async_reader():
    """
    Shutdown the global AsyncExpertReader and release resources.
    """
    global _global_async_reader, _global_async_readers
    _global_async_reader = None
    _global_async_readers = []


def is_async_reader_initialized():
    """
    Check if the global AsyncExpertReader is initialized.

    Returns:
        bool: True if initialized, False otherwise
    """
    return _global_async_reader is not None or bool(_global_async_readers)
