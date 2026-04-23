from datetime import datetime
import hashlib
import hmac
import io
import os
import pickle
import secrets
from typing import Optional
import zmq
import threading
import torch
import torch.multiprocessing as mp
import sys
current_file_path = os.path.abspath(__file__)
import argparse
from safetensors.torch import save, load as st_load
from ktransformers.server.balance_serve.settings import sched_ext, create_sched_settings, create_sched_settings_qwen2moe, create_sched_settings_qwen3moe


# ---------------------------------------------------------------------------
# HMAC authentication
# ---------------------------------------------------------------------------
# The parent process (balance_serve.py) MUST generate a secret and set
# KTRANSFORMERS_RPC_SECRET in the environment before spawning this process.
# If unset, a random key is generated (works only for single-process or
# fork-inherited setups).

def _get_rpc_secret() -> bytes:
    env = os.environ.get("KTRANSFORMERS_RPC_SECRET", "")
    if env:
        return env.encode()
    secret = secrets.token_bytes(32)
    os.environ["KTRANSFORMERS_RPC_SECRET"] = secret.hex()
    return secret


_RPC_SECRET = _get_rpc_secret()


def _sign(data: bytes) -> bytes:
    return hmac.new(_RPC_SECRET, data, hashlib.sha256).digest()


def _verify(data: bytes, sig: bytes) -> bool:
    return hmac.compare_digest(_sign(data), sig)


# ---------------------------------------------------------------------------
# Restricted unpickler - only allow known safe types
# ---------------------------------------------------------------------------
# pickle is still required for the C++ scheduler extension objects
# (QueryAdd, QueryUpdate, BatchQueryTodo, etc.) that are not
# JSON-serializable. This restricted unpickler ensures only explicitly
# allowed types can be deserialized, preventing arbitrary code execution.

_ALLOWED_MODULES = {
    "builtins": {"dict", "list", "tuple", "set", "frozenset", "int", "float",
                 "str", "bytes", "bool", "NoneType", "complex", "range",
                 "slice", "type"},
    "collections": {"OrderedDict", "defaultdict"},
    "datetime": {"datetime", "timedelta", "date", "time"},
    "ktransformers.server.balance_serve.settings": {"*"},
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        allowed = _ALLOWED_MODULES.get(module)
        if allowed is not None and ("*" in allowed or name in allowed):
            return super().find_class(module, name)
        # Also allow sched_ext types (C++ extension objects)
        if module.startswith("ktransformers"):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Restricted unpickler refused {module}.{name}"
        )


def _safe_loads(data: bytes):
    return RestrictedUnpickler(io.BytesIO(data)).load()


# ---------------------------------------------------------------------------
# Safetensors helpers for KV cache
# ---------------------------------------------------------------------------

def _serialize_tensors(tensor_dict: dict) -> bytes:
    buf = io.BytesIO()
    save(tensor_dict, buf)
    return buf.getvalue()


def _deserialize_tensors(data: bytes) -> dict:
    return st_load(data)


# ---------------------------------------------------------------------------

if mp.get_start_method(allow_none=True) is None:
    print('set start method')
    mp.set_start_method('spawn')
else:
    print(f'start method already set to {mp.get_start_method(allow_none=True)}')


class SchedulerServer:
    def __init__(self, settings, main_args):
        self.sched = sched_ext.create_scheduler(settings)

        self.context = zmq.Context()
        self.frontend = self.context.socket(zmq.ROUTER)

        bind_addr = getattr(main_args, 'sched_bind', '127.0.0.1')
        print(f"sched zmq rpc server on {bind_addr}:{main_args.sched_port}")
        self.frontend.bind(f"tcp://{bind_addr}:{main_args.sched_port}")

        self.backend = self.context.socket(zmq.DEALER)
        self.backend.bind("inproc://backend")

    def run_scheduler(self):
        self.sched.run()

    def stop_scheduler(self):
        self.sched.stop()

    def start_proxy(self):
        zmq.proxy(self.frontend, self.backend)

    def _send(self, worker, response: dict, tensor_data: bytes = b""):
        payload = pickle.dumps(response)
        sig = _sign(payload + tensor_data)
        worker.send_multipart([sig, payload, tensor_data])

    def _recv(self, worker) -> dict:
        parts = worker.recv_multipart()
        if len(parts) != 3:
            raise ValueError("Invalid message frame")
        sig, payload, tensor_data = parts
        if not _verify(payload + tensor_data, sig):
            raise ValueError("HMAC verification failed - unauthorized message")
        return _safe_loads(payload)

    def worker_routine(self):
        worker = self.context.socket(zmq.REP)
        worker.connect("inproc://backend")
        while True:
            try:
                data = self._recv(worker)

                method = data.get('method')
                params = data.get('params', {})

                if method == 'add_query':
                    query_add = params.get('query')
                    query_id = self.sched.add_query(query_add)
                    response = {'status': 'ok', 'query_id': query_id}
                    self._send(worker, response)

                elif method == 'cancel_query':
                    query_id = params.get('query_id')
                    self.sched.cancel(query_id)
                    response = {'status': 'ok'}
                    self._send(worker, response)

                elif method == 'update_last_batch':
                    updates = params.get('updates')
                    batch_todo = self.sched.update_last_batch(updates)
                    response = {'status': 'ok', 'batch_todo': batch_todo}
                    self._send(worker, response)

                elif method == 'get_inference_context':
                    inference_context = self.sched.get_inference_context()
                    print("Serializing KVCache with safetensors")

                    tensors = {}
                    for i, t in enumerate(inference_context.k_cache):
                        tensors[f"k_cache_{i}"] = t
                    for i, t in enumerate(inference_context.v_cache):
                        tensors[f"v_cache_{i}"] = t

                    tensor_bytes = _serialize_tensors(tensors)
                    k_count = len(inference_context.k_cache)
                    v_count = len(inference_context.v_cache)
                    response = {
                        'status': 'ok',
                        'k_cache_count': k_count,
                        'v_cache_count': v_count,
                    }
                    self._send(worker, response, tensor_bytes)

                else:
                    response = {'status': 'error', 'message': 'Unknown method'}
                    self._send(worker, response)

            except Exception as e:
                try:
                    response = {'status': 'error', 'message': str(e)}
                    self._send(worker, response)
                except Exception:
                    pass

    def start_rpc_service(self):
        try:
            print("Scheduler RPC service is running...")

            threading.Thread(target=self.run_scheduler, daemon=True).start()

            for _ in range(10):
                threading.Thread(target=self.worker_routine, daemon=True).start()

            self.start_proxy()

        except KeyboardInterrupt:
            print("Shutting down scheduler RPC service...")
            self.stop_rpc_service()

    def stop_rpc_service(self):
        self.stop_scheduler()
        self.frontend.close()
        self.backend.close()
        self.context.term()

def start_server(settings, main_args):
    server = SchedulerServer(settings, main_args)
    server.start_rpc_service()


class SchedulerClient:
    def __init__(self, sched_port):
        address = f'tcp://localhost:{sched_port}'
        self.address = address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.address)
        print(f"Connected to server at {self.address}")

    def __del__(self):
        self.socket.close()
        self.context.term()

    def _send(self, request: dict):
        payload = pickle.dumps(request)
        sig = _sign(payload + b"")
        self.socket.send_multipart([sig, payload, b""])

    def _recv(self) -> tuple:
        parts = self.socket.recv_multipart()
        if len(parts) != 3:
            raise ValueError("Invalid message frame")
        sig, payload, tensor_data = parts
        if not _verify(payload + tensor_data, sig):
            raise ValueError("HMAC verification failed - unauthorized message")
        return _safe_loads(payload), tensor_data

    def send_request(self, method, params=None):
        if params is None:
            params = {}
        request = {
            'method': method,
            'params': params
        }
        self._send(request)
        response, _ = self._recv()
        if response.get('status') == 'ok':
            return response
        else:
            raise Exception(f"Error from server: {response.get('message')}")

    def add_query(self, query):
        response = self.send_request('add_query', {'query': query})
        return response.get('query_id')

    def cancel_query(self, query_id):
        self.send_request('cancel_query', {'query_id': query_id})

    def update_last_batch(self, updates):
        response = self.send_request('update_last_batch', {'updates': updates})
        return response.get('batch_todo')

    def rebuild_inferece_context(self, response=None, tensor_data=None):
        if tensor_data is None:
            raise ValueError("No tensor data received")
        tensors = _deserialize_tensors(tensor_data)

        inference_context = sched_ext.InferenceContext()
        print('Rebuilding kvcache from safetensors')

        k_count = response.get('k_cache_count', 0)
        v_count = response.get('v_cache_count', 0)
        inference_context.k_cache = [tensors[f"k_cache_{i}"] for i in range(k_count)]
        inference_context.v_cache = [tensors[f"v_cache_{i}"] for i in range(v_count)]
        return inference_context

    def get_inference_context_raw(self):
        request = {
            'method': 'get_inference_context',
            'params': {}
        }
        self._send(request)
        response, tensor_data = self._recv()
        if response.get('status') == 'ok':
            return response, tensor_data
        else:
            raise Exception(f"Error from server: {response.get('message')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "rb") as f:
        main_args = _safe_loads(f.read())
    if main_args.architectures == "Qwen2MoeForCausalLM":
        settings = create_sched_settings_qwen2moe(main_args)
    elif main_args.architectures == "Qwen3MoeForCausalLM":
        settings = create_sched_settings_qwen3moe(main_args)
    else:
        settings = create_sched_settings(main_args)
    start_server(settings, main_args)
