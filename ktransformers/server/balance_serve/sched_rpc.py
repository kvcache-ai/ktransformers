from datetime import datetime
import os
from typing import Optional
import zmq
import pickle
import threading
import torch.multiprocessing as mp
import sys
current_file_path = os.path.abspath(__file__)
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import pickle
import argparse
from ktransformers.server.balance_serve.settings import sched_ext, create_sched_settings, create_sched_settings_qwen2moe, create_sched_settings_qwen3moe



if mp.get_start_method(allow_none=True) is None:
    print('set start method')
    mp.set_start_method('spawn')
else:
    print(f'start method already set to {mp.get_start_method(allow_none=True)}')


class SchedulerServer:
    def __init__(self, settings, main_args):
        # 创建 Scheduler 实例并初始化
        self.sched = sched_ext.create_scheduler(settings)
    
        # 初始化 ZeroMQ 上下文和套接字
        self.context = zmq.Context()
        self.frontend = self.context.socket(zmq.ROUTER)
        print(f"sched zmq rpc server on port {main_args.sched_port}")
        self.frontend.bind(f"tcp://*:{main_args.sched_port}") 

        # 创建内部的 DEALER 套接字，用于与工作线程通信
        self.backend = self.context.socket(zmq.DEALER)
        self.backend.bind("inproc://backend")

    # 启动调度器
    def run_scheduler(self):
        self.sched.run()

    # 停止调度器
    def stop_scheduler(self):
        self.sched.stop()

    # 处理客户端请求
    def start_proxy(self):
        # 使用 ZMQ 的内置代理，将前端请求分发给后端工作线程
        zmq.proxy(self.frontend, self.backend)

    # 工作线程处理请求
    def worker_routine(self):
        worker = self.context.socket(zmq.REP)
        worker.connect("inproc://backend")
        while True:
            try:
                # 接收客户端请求
                message = worker.recv()
                data = pickle.loads(message)

                method = data.get('method')
                params = data.get('params', {})
                # print(f"Received request: {method}")

                if method == 'add_query':
                    query_add = params.get('query')  # 直接是一个 QueryAdd 对象
                    # 添加查询
                    query_id = self.sched.add_query(query_add)
                    # 发送响应
                    response = {'status': 'ok', 'query_id': query_id}
                    worker.send(pickle.dumps(response))

                elif method == 'cancel_query':
                    query_id = params.get('query_id')
                    # 假设您的 Scheduler 类实现了 cancel 方法
                    self.sched.cancel(query_id)
                    response = {'status': 'ok'}
                    worker.send(pickle.dumps(response))

                elif method == 'update_last_batch':
                    updates = params.get('updates')  # 直接是一个列表，包含 QueryUpdate 对象

                    # 更新最后一个批次
                    batch_todo = self.sched.update_last_batch(updates)

                    # 直接发送 batch_todo 对象
                    response = {'status': 'ok', 'batch_todo': batch_todo}
                    # print (batch_todo.query_lengths, batch_todo.query_ids)
                    worker.send(pickle.dumps(response))

                elif method == 'get_inference_context':
                    inference_context = self.sched.get_inference_context()
                    data = {
                        "k_cache":inference_context.k_cache,
                        "v_cache":inference_context.v_cache
                    }
                    print(f"Serializing KVCache")
                    data["k_cache"] = [mp.reductions.reduce_tensor(t) for t in data['k_cache']]
                    data["v_cache"] = [mp.reductions.reduce_tensor(t) for t in data['v_cache']]
                    # print(data)
                    response = {'status': 'ok', 'inference_context': data}

                    worker.send(pickle.dumps(response))
                    # response['inference_context'].k_cache[0][0, 0, 0, 0, 0] = 1 
                    # print("k_cache update")

                else:
                    # 未知方法
                    response = {'status': 'error', 'message': 'Unknown method'}
                    worker.send(pickle.dumps(response))

            except Exception as e:
                # 处理异常并发送错误响应
                response = {'status': 'error', 'message': str(e)}
                worker.send(pickle.dumps(response))

    # 启动 RPC 服务
    def start_rpc_service(self):
        try:
            print("Scheduler RPC service is running...")

            # 在单独的线程中运行调度器
            threading.Thread(target=self.run_scheduler, daemon=True).start()

            # 启动工作线程
            for _ in range(10):  # 根据需要调整线程数
                threading.Thread(target=self.worker_routine, daemon=True).start()

            # 启动代理，开始监听请求
            self.start_proxy()

        except KeyboardInterrupt:
            print("Shutting down scheduler RPC service...")
            self.stop_rpc_service()

    # 停止 RPC 服务
    def stop_rpc_service(self):
        self.stop_scheduler()
        self.frontend.close()
        self.backend.close()
        self.context.term()

def start_server(settings, main_args):
    server = SchedulerServer(settings, main_args)
    server.start_rpc_service()


# Add async client for webserver
class SchedulerClient:
    def __init__(self, sched_port):
        address=f'tcp://localhost:{sched_port}'
        self.address = address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.address)
        print(f"Connected to server at {self.address}")
    
    def __del__(self):
        self.socket.close()
        self.context.term()
    
    def send_request(self, method, params=None):
        if params is None:
            params = {}
        request = {
            'method': method,
            'params': params
        }
        # print(f'send request {request}')
        self.socket.send(pickle.dumps(request))
        response = self.socket.recv()
        # print(response)
        response = pickle.loads(response)
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
        # print(f"update_last_batch response {response}")
        return response.get('batch_todo')
    
    def rebuild_inferece_context(self,response):
        data = response.get('inference_context')
        inference_context = sched_ext.InferenceContext()
        print('Rebuilding kvcache')
        inference_context.k_cache = [fn(*args) for fn,args in data['k_cache']]
        inference_context.v_cache = [fn(*args) for fn,args in data['v_cache']]
        return inference_context

    def get_inference_context_raw(self):
        response = self.send_request('get_inference_context')
        return response
       


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "rb") as f:
        main_args = pickle.load(f)
    if main_args.architectures == "Qwen2MoeForCausalLM": 
        settings = create_sched_settings_qwen2moe(main_args)
    elif main_args.architectures == "Qwen3MoeForCausalLM":
        settings = create_sched_settings_qwen3moe(main_args)
    else:
        settings = create_sched_settings(main_args)
    start_server(settings, main_args)
