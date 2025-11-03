# test bench_moe_kernel_tiling.py
kernel_tiling:
	python3 bench_moe_kernel_tiling.py \
	--hidden_size 7168 \
	--intermediate_size 2048 \
	--num_experts_per_tok 8 \
	--expert_num 256 \
	--max_len 51200 \
	--layer_num 1 \
	--qlen 1024 \
	--quant int8 \
	--warm_up_iter 500 \
	--test_iter 1000 \
	--threads 160 \
	--m_block 320 \
	
# 	--n_block_up_gate 256 \
# 	--n_block_down 128 \
# 	--n_block_up_gate_prefi 256 \
# 	--n_block_down_prefi 128 \

# 	--n_block_up_gate 256 \
# 	--n_block_down 512 \