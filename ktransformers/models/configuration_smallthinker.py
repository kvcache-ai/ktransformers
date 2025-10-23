# coding=utf-8
from transformers.configuration_utils import PretrainedConfig

class SmallthinkerConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`SmallthinkerModel`]. 
    It is used to instantiate a Smallthinker model according to the specified arguments, defining the model architecture. 
    The default values for each of the parameters are the same as the ones used in the original Smallthinker 4B model.

    General configs:
    - model_type: "smallthinker"
    - model_name
    - num_hidden_layers
    - hidden_size

    Tokenizer configs:
    - pad_token_id
    - bos_token_id
    - eos_token_id

    Embedding configs:
    - vocab_size

    RMSNorm configs:
    - rms_norm_eps

    Attention configs:
    - num_attention_heads
    - num_key_value_heads
    - head_dim
    - use_cache
    - use_qk_norm
    - rope_layout: array of 0 or 1s, 0 for nope, 1 for rope
    - rope_theta
    - max_position_embeddings
    - sliding_window_layout: array of 0 or 1s, 0 for normal attention, 1 for SWA
    - sliding_window_size

    General FFN configs:
    - moe_layer_layout: array of 0 or 1s, 0 for dense layer, 1 for MoE layer
    
    Dense FFN configs:
    - dense_ffn_hidden_size

    MoE FFN configs:
    - moe_num_primary_experts
    - moe_shared_primary_experts
    - moe_ffn_hidden_size
    - moe_enable_early_router: Use attention output as router input if true
    - moe_primary_router_use_sigmoid: Use normalized sigmoid 
    - moe_num_active_primary_experts
    - moe_enable_secondary_experts
    - moe_num_secondary_experts
    - moe_secondary_expert_size

    LM Head configs:
    - tie_word_embeddings

    Visibility configs:
    - profile_sparsity

    Other configs:
    - initializer_range
    """
    def __init__(self,
        model_type = "smallthinker",
        model_name="smallthinker_4b_base",
        num_hidden_layers=32,
        hidden_size=1536,
        pad_token_id=None,
        bos_token_id=151643,
        eos_token_id=[151643,151645],
        vocab_size=151936,
        rms_norm_eps=1e-6,
        num_attention_heads=12,
        num_key_value_heads=2,
        head_dim=128,
        use_cache=True,
        use_qk_norm=False,
        rope_layout=[1]*32,
        rope_theta=1e6,
        max_position_embeddings=4096 * 32,
        sliding_window_layout=[0]*32,
        sliding_window_size=4096,
        moe_layer_layout=[1]*32,
        dense_ffn_hidden_size=4096,
        moe_num_primary_experts=32,
        moe_shared_primary_experts=0,
        moe_ffn_hidden_size=768,
        moe_enable_early_router=True,
        moe_primary_router_apply_softmax=False,
        moe_num_active_primary_experts=4,
        moe_enable_secondary_experts=False,
        moe_num_secondary_experts=0,
        moe_secondary_expert_size=0,
        tie_word_embeddings=True,
        initializer_range=0.02,
        **kwargs,
    ):
        moe_layer_layout = [1]*num_hidden_layers
        # Configuration sanitizers
        assert num_attention_heads % num_key_value_heads == 0,      "[Smallthinker config sanitizer] num_attention_heads must be divisible by num_key_value_heads"
        assert len(rope_layout) == num_hidden_layers,               "[Smallthinker config sanitizer] rope_layout must have the same length as num_hidden_layers"
        assert len(sliding_window_layout) == num_hidden_layers,     "[Smallthinker config sanitizer] sliding_window_layout must have the same length as num_hidden_layers"
        assert len(moe_layer_layout) == num_hidden_layers,          "[Smallthinker config sanitizer] moe_layer_layout must have the same length as num_hidden_layers"

        if any(moe_layer_layout):
            assert moe_num_primary_experts != 0,                    "[Smallthinker config sanitizer] moe_num_primary_experts must be set non-zero if there is any MoE layer"
            assert moe_ffn_hidden_size != 0,                        "[Smallthinker config sanitizer] moe_ffn_hidden_size must be set non-zero if there is any MoE layer"
            assert moe_num_active_primary_experts != 0,             "[Smallthinker config sanitizer] moe_num_active_primary_experts must be set non-zero if there is any MoE layer"
            if moe_enable_secondary_experts:
                assert moe_num_secondary_experts != 0,              "[Smallthinker config sanitizer] moe_num_secondary_experts must be set non-zero if moe_enable_secondary_experts is True"
                assert moe_secondary_expert_size != 0,              "[Smallthinker config sanitizer] moe_secondary_expert_size must be set non-zero if moe_enable_secondary_experts is True"
                assert moe_num_secondary_experts * moe_secondary_expert_size == moe_ffn_hidden_size, "[Smallthinker config sanitizer] moe_num_secondary_experts * moe_secondary_expert_size must equal moe_ffn_hidden_size"

        if not all(moe_layer_layout):
            assert dense_ffn_hidden_size != 0,                      "[Smallthinker config sanitizer] dense_ffn_hidden_size must be set non-zero if there is any dense FFN layer"

        # General configs
        self.model_type = model_type
        self.model_name = model_name
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

        # Tokenizer configs
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # Embedding configs
        self.vocab_size = vocab_size

        # RMSNorm configs
        self.rms_norm_eps = rms_norm_eps

        # Attention configs
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.use_cache = use_cache
        self.use_qk_norm = use_qk_norm
        self.rope_layout = rope_layout
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window_layout = sliding_window_layout
        self.sliding_window_size = sliding_window_size

        # General FFN configs
        self.moe_layer_layout = moe_layer_layout

        # Dense FFN configs
        self.dense_ffn_hidden_size = dense_ffn_hidden_size

        # MoE FFN configs
        self.moe_num_primary_experts = moe_num_primary_experts
        self.moe_shared_primary_experts = moe_shared_primary_experts
        self.moe_ffn_hidden_size = moe_ffn_hidden_size
        self.num_experts_per_tok = moe_num_active_primary_experts
        self.moe_intermediate_size = moe_ffn_hidden_size
        self.moe_enable_early_router = moe_enable_early_router
        self.moe_primary_router_apply_softmax = moe_primary_router_apply_softmax
        self.moe_num_active_primary_experts = moe_num_active_primary_experts
        self.moe_enable_secondary_experts = moe_enable_secondary_experts
        self.moe_num_secondary_experts = moe_num_secondary_experts
        self.moe_secondary_expert_size = moe_secondary_expert_size

        # Logging configs
        # self.output_router_logits = False

        # Other configs
        self.initializer_range = initializer_range

        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

        self._attn_implementation = "eager" # SDPA is not allowed for now

        # if self._attn_implementation != "flash_attention_2":
        #     raise NotImplementedError("SDPA impl is buggy for now. NEVER TRY TO USE IT.")
        
__all__ = ["SmallthinkerConfig"]
