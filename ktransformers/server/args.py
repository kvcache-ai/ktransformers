import argparse
from ktransformers.server.backend.args import ConfigArgs, default_args


class ArgumentParser:
    def __init__(self, cfg):
        self.cfg = cfg

    def parse_args(self):
        parser = argparse.ArgumentParser(prog="kvcache.ai", description="Ktransformers")
        parser.add_argument("--host", type=str, default=self.cfg.server_ip)
        parser.add_argument("--port", type=int, default=self.cfg.server_port)
        parser.add_argument("--api_key", type=str, default=self.cfg.api_key)
        parser.add_argument("--ssl_keyfile", type=str)
        parser.add_argument("--ssl_certfile", type=str)
        parser.add_argument("--web", type=bool, default=self.cfg.mount_web)
        parser.add_argument("--model_name", type=str, default=self.cfg.model_name)
        parser.add_argument("--model_dir", type=str)
        parser.add_argument("--model_path", type=str)
        parser.add_argument(
            "--device", type=str, default=self.cfg.model_device, help="Warning: Abandoning this parameter"
        )
        parser.add_argument("--gguf_path", type=str, default=self.cfg.gguf_path)
        parser.add_argument("--optimize_config_path", default=self.cfg.optimize_config_path, type=str, required=False)
        parser.add_argument("--cpu_infer", type=int, default=self.cfg.cpu_infer)
        parser.add_argument("--type", type=str, default=self.cfg.backend_type)
        parser.add_argument("--chunk_prefill_size", type=int, default=8192)

        # model configs
        # parser.add_argument("--model_cache_lens", type=int, default=self.cfg.cache_lens)  # int?
        parser.add_argument("--paged", type=bool, default=self.cfg.paged)
        parser.add_argument("--total_context", type=int, default=self.cfg.total_context)
        parser.add_argument("--max_batch_size", type=int, default=self.cfg.max_batch_size)
        parser.add_argument("--max_new_tokens", type=int, default=self.cfg.max_new_tokens)
        parser.add_argument("--json_mode", type=bool, default=self.cfg.json_mode)
        parser.add_argument("--healing", type=bool, default=self.cfg.healing)
        parser.add_argument("--ban_strings", type=list, default=self.cfg.ban_strings, required=False)
        parser.add_argument("--gpu_split", type=str, default=self.cfg.gpu_split, required=False)
        parser.add_argument("--length", type=int, default=self.cfg.length, required=False)
        parser.add_argument("--rope_scale", type=float, default=self.cfg.rope_scale, required=False)
        parser.add_argument("--rope_alpha", type=float, default=self.cfg.rope_alpha, required=False)
        parser.add_argument("--no_flash_attn", type=bool, default=self.cfg.no_flash_attn)
        parser.add_argument("--low_mem", type=bool, default=self.cfg.low_mem)
        parser.add_argument("--experts_per_token", type=int, default=self.cfg.experts_per_token, required=False)
        parser.add_argument("--load_q4", type=bool, default=self.cfg.load_q4)
        parser.add_argument("--fast_safetensors", type=bool, default=self.cfg.fast_safetensors)
        parser.add_argument("--draft_model_dir", type=str, default=self.cfg.draft_model_dir, required=False)
        parser.add_argument("--no_draft_scale", type=bool, default=self.cfg.no_draft_scale)
        parser.add_argument("--modes", type=bool, default=self.cfg.modes)
        parser.add_argument("--mode", type=str, default=self.cfg.mode)
        parser.add_argument("--username", type=str, default=self.cfg.username)
        parser.add_argument("--botname", type=str, default=self.cfg.botname)
        parser.add_argument("--system_prompt", type=str, default=self.cfg.system_prompt, required=False)
        parser.add_argument("--temperature", type=float, default=self.cfg.temperature)
        parser.add_argument("--smoothing_factor", type=float, default=self.cfg.smoothing_factor)
        parser.add_argument("--dynamic_temperature", type=str, default=self.cfg.dynamic_temperature, required=False)
        parser.add_argument("--top_k", type=int, default=self.cfg.top_k)
        parser.add_argument("--top_p", type=float, default=self.cfg.top_p)
        parser.add_argument("--top_a", type=float, default=self.cfg.top_a)
        parser.add_argument("--skew", type=float, default=self.cfg.skew)
        parser.add_argument("--typical", type=float, default=self.cfg.typical)
        parser.add_argument("--repetition_penalty", type=float, default=self.cfg.repetition_penalty)
        parser.add_argument("--frequency_penalty", type=float, default=self.cfg.frequency_penalty)
        parser.add_argument("--presence_penalty", type=float, default=self.cfg.presence_penalty)
        parser.add_argument("--max_response_tokens", type=int, default=self.cfg.max_response_tokens)
        parser.add_argument("--response_chunk", type=int, default=self.cfg.response_chunk)
        parser.add_argument("--no_code_formatting", type=bool, default=self.cfg.no_code_formatting)
        parser.add_argument("--cache_8bit", type=bool, default=self.cfg.cache_8bit)
        parser.add_argument("--cache_q4", type=bool, default=self.cfg.cache_q4)
        parser.add_argument("--ngram_decoding", type=bool, default=self.cfg.ngram_decoding)
        parser.add_argument("--print_timings", type=bool, default=self.cfg.print_timings)
        parser.add_argument("--amnesia", type=bool, default=self.cfg.amnesia)
        parser.add_argument("--batch_size", type=int, default=self.cfg.batch_size)
        parser.add_argument("--cache_lens", type=int, default=self.cfg.cache_lens)

        # log configs
        # log level: debug, info, warn, error, crit
        parser.add_argument("--log_dir", type=str, default=self.cfg.log_dir)
        parser.add_argument("--log_file", type=str, default=self.cfg.log_file)
        parser.add_argument("--log_level", type=str, default=self.cfg.log_level)
        parser.add_argument("--backup_count", type=int, default=self.cfg.backup_count)

        # db configs
        parser.add_argument("--db_type", type=str, default=self.cfg.db_type)
        parser.add_argument("--db_host", type=str, default=self.cfg.db_host)
        parser.add_argument("--db_port", type=str, default=self.cfg.db_port)
        parser.add_argument("--db_name", type=str, default=self.cfg.db_name)
        parser.add_argument("--db_pool_size", type=int, default=self.cfg.db_pool_size)
        parser.add_argument("--db_database", type=str, default=self.cfg.db_database)

        # user config
        parser.add_argument("--user_secret_key", type=str, default=self.cfg.user_secret_key)
        parser.add_argument("--user_algorithm", type=str, default=self.cfg.user_algorithm)
        parser.add_argument("--force_think", action=argparse.BooleanOptionalAction, type=bool, default=self.cfg.user_force_think)
        parser.add_argument("--use_cuda_graph", action=argparse.BooleanOptionalAction, type=bool, default=self.cfg.use_cuda_graph)

        # web config
        parser.add_argument("--web_cross_domain", type=bool, default=self.cfg.web_cross_domain)

        # file config
        parser.add_argument("--file_upload_dir", type=str, default=self.cfg.file_upload_dir)
        parser.add_argument("--assistant_store_dir", type=str, default=self.cfg.assistant_store_dir)
        # local chat
        parser.add_argument("--prompt_file", type=str, default=self.cfg.prompt_file)

        args = parser.parse_args()
        if (args.model_dir is not None or args.model_path is not None):
            if (args.model_path is not None):
                # if pass model_dir and model_path, we use model_path
                args.model_dir = args.model_path
            else:
                # if only pass model_dir, we use model_dir
                args.model_path = args.model_dir
        else:
            args.model_dir = self.cfg.model_dir
            args.model_path = self.cfg.model_path
        # set config from args
        for key, value in vars(args).items():
            if value is not None and hasattr(self.cfg, key):
                setattr(self.cfg, key, value)
        # we add the name not match args individually
        self.cfg.model_device = args.device
        self.cfg.mount_web = args.web
        self.cfg.server_ip = args.host
        self.cfg.server_port = args.port
        self.cfg.backend_type = args.type
        self.cfg.user_force_think = args.force_think
        return args
