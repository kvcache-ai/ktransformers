from pydantic import BaseModel,Field
from typing import Optional
from ktransformers.server.config.config import Config


class ConfigArgs(BaseModel):
    model_name : Optional[str] = Field(..., description="Model name")
    model_dir: Optional[str] = Field(..., description="Path to model directory")
    optimize_config_path: Optional[str] = Field('./KTransformers/optimize_config/DeepSeek-V2-Chat.json', description="Path of your optimize config json file")
    gguf_path: Optional[str] = Field('/models/DeepSeek-Coder-V2-Instruct-GGUF/DeepSeek-Coder-V2-Instruct-Q4_K_M.gguf', description="Path of your gguf file")
    class Config:
        protected_namespaces = ()

    paged : bool = Field(True,description='Wether to use paged attention kv cache')

    # total_context: int = Field(16384, description="Total number of tokens to allocate space for. This is not the max_seq_len supported by the model but the total to distribute dynamically over however many jobs are active at once")
    total_context: int = Field(2**18, description="Total number of tokens to allocate space for. This is not the max_seq_len supported by the model but the total to distribute dynamically over however many jobs are active at once")
    max_batch_size: int = Field(20 if paged else 1, description="Max number of batches to run at once, assuming the sequences will fit within total_context")
    max_chunk_size: int = Field(2048, description="Max chunk size. Determines the size of prefill operations. Can be reduced to reduce pauses whenever a new job is started, but at the expense of overall prompt ingestion speed")
    max_new_tokens: int = Field(500, description="Max new tokens per completion. For this example applies to all jobs")
    json_mode: bool = Field(False, description="Use LMFE to constrain the output to JSON format. See schema and details below")
    healing: bool = Field(False, description="Demonstrate token healing")
    ban_strings: Optional[list] = Field(None, description="Ban some phrases maybe")

    gpu_split: Optional[str] = Field(None, description='"auto", or VRAM allocation per GPU in GB')
    length: Optional[int] = Field(None, description="Maximum sequence length")
    rope_scale: Optional[float] = Field(None, description="RoPE scaling factor")
    rope_alpha: Optional[float] = Field(None, description="RoPE alpha value (NTK)")
    no_flash_attn: bool = Field(False, description="Disable Flash Attention")
    low_mem: bool = Field(
        False,
        description="Enable VRAM optimizations, potentially trading off speed",
    )
    experts_per_token: Optional[int] = Field(
        None,
        description="Override MoE model's default number of experts per token",
    )
    load_q4: bool = Field(False, description="Load weights in Q4 mode")
    fast_safetensors: bool = Field(
        False,
        description="Optimized safetensors loading with direct I/O (experimental!)",
    )
    draft_model_dir: Optional[str] = Field(None, description="Path to draft model directory")
    no_draft_scale: bool = Field(
        False,
        description="If draft model has smaller context size than model, don't apply alpha (NTK) scaling to extend it",
    )
    modes: bool = Field(False, description="List available modes and exit.")
    mode: str = Field(
        "llama",
        description="Chat mode. Use llama for Llama 1/2 chat finetunes.",
    )
    username: str = Field("User", description="Username when using raw chat mode")
    botname: str = Field("Chatbort", description="Bot name when using raw chat mode")
    system_prompt: Optional[str] = Field(None, description="Use custom system prompt")
    temperature: float = Field(0.95, description="Sampler temperature, default = 0.95 (1 to disable)")
    smoothing_factor: float = Field(0.0, description="Smoothing Factor, default = 0.0 (0 to disable)")
    dynamic_temperature: Optional[str] = Field(
        None,
        description="Dynamic temperature min,max,exponent, e.g. -dyntemp 0.2,1.5,1",
    )
    top_k: int = Field(50, description="Sampler top-K, default = 50 (0 to disable)")
    top_p: float = Field(0.8, description="Sampler top-P, default = 0.8 (0 to disable)")
    top_a: float = Field(0.0, description="Sampler top-A, default = 0.0 (0 to disable)")
    skew: float = Field(0.0, description="Skew sampling, default = 0.0 (0 to disable)")
    typical: float = Field(
        0.0,
        description="Sampler typical threshold, default = 0.0 (0 to disable)",
    )
    repetition_penalty: float = Field(
        1.01,
        description="Sampler repetition penalty, default = 1.01 (1 to disable)",
    )
    frequency_penalty: float = Field(
        0.0,
        description="Sampler frequency penalty, default = 0.0 (0 to disable)",
    )
    presence_penalty: float = Field(
        0.0,
        description="Sampler presence penalty, default = 0.0 (0 to disable)",
    )
    max_response_tokens: int = Field(300, description="Max tokens per response, default = 1000")
    response_chunk: int = Field(250, description="Space to reserve in context for reply, default = 250")
    no_code_formatting: bool = Field(False, description="Disable code formatting/syntax highlighting")
    cache_8bit: bool = Field(False, description="Use 8-bit (FP8) cache")
    cache_q4: bool = Field(True, description="Use Q4 cache")
    ngram_decoding: bool = Field(False, description="Use n-gram speculative decoding")
    print_timings: bool = Field(False, description="Output timings after each prompt")
    amnesia: bool = Field(False, description="Forget context after every response")

    # for transformers
    batch_size :int = Field(1,description="Batch Size")
    cache_lens:int = Field(4096, description="Cache lens for transformers static cache")
    device:str = Field('cuda:2',description="device")

cfg = Config()
default_args = ConfigArgs(model_name=cfg.model_name,model_dir=cfg.model_path)
