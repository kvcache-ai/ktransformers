DECODE_LENGTH="512"
NUM_TRIALS="3"
OUTPUT_FILE="decode_perf.jsonl"

eval_decode() {
    local model_path="$1"
    local config_id="$2"

    final_json=$(../../baseline-custom-llama.cpp/build/bin/llama-bench -m "$model_path" -ngl 1000 -p 0 -n "$DECODE_LENGTH" -r "$NUM_TRIALS" -nkvo 1 --numa distribute -mmp 0 -o jsonl | \
        tee /dev/stderr | \
        grep -E '^\s*\{.*\}\s*$' | \
        jq -c 'select(.avg_ts != null) | .avg_ts = (.avg_ts | tonumber | .*100 | round / 100)' | \
        jq -c --arg config "$config_id" \
            '{config_id: $config, result: .avg_ts}'
    )

    echo "$final_json" >> "$OUTPUT_FILE"
    echo "✅ Results saved to $OUTPUT_FILE"
}

eval_decode "$QW2_GGUF_F16" llama.cpp_qw2_fp16
eval_decode "$QW2_GGUF_Q8_0" llama.cpp_qw2_int8