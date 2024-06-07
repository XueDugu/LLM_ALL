CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir '/mnt/inside_15T/PPG_dataset/aibot/ypj_model/out/qwen1half-0_5b-chat/v14-20240515-155739/checkpoint-1560' \
    --merge_lora true --quant_bits 4 \
    --load_dataset_config true --quant_method awq