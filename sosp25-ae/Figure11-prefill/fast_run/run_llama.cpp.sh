PROMPT_LENGTH_LIST="32,64,128,256,512,1024,2048,4096,8192"
NUM_TRIALS="3"
OUTPUT_FILE="prefill_perf.jsonl"

eval_prefill() {
    local model_path="$1"
    local config_id="$2"
    
    final_json=$(../../baseline-custom-llama.cpp/build/bin/llama-bench -m "$model_path" -ngl 1000 -p "$PROMPT_LENGTH_LIST" -n 0 -r "$NUM_TRIALS" -nkvo 1 --numa distribute -mmp 0 -o jsonl | \
        tee /dev/stderr | \
        grep -E '^\s*\{.*\}\s*$' | \
        jq -c 'select(.avg_ts != null) | .avg_ts = (.avg_ts | tonumber | .*100 | round / 100)' | \
        jq -sc --arg config "$config_id" '
                {
                    config_id: $config,
                    results: (map({(.n_prompt|tostring): .avg_ts}) | add)
                }
        ')
    
    echo "$final_json" >> "$OUTPUT_FILE"
    echo "✅ Results saved to $OUTPUT_FILE"
}

eval_prefill "$QW2_GGUF_F16" llama.cpp_qw2_fp16
eval_prefill "$QW2_GGUF_Q8_0" llama.cpp_qw2_int8