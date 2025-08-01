
python ../../baseline-fiddler/eval_decode.py \
    --model_path "$DS3_SAFETENSORS" \
    --config_id fiddler_ds3_bf16

python ../../baseline-fiddler/eval_decode.py \
    --model_path "$DS2_SAFETENSORS" \
    --config_id fiddler_ds2_bf16

python ../../baseline-fiddler/eval_decode.py \
    --model_path "$QW2_SAFETENSORS" \
    --config_id fiddler_qw2_bf16
