# Experimental environment: A100
# 26GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import DatasetName, ModelType, SftArguments, sft_main

sft_args = SftArguments(
    model_type=ModelType.qwen1half_0_5b_chat,
    # dataset=[DatasetName.ms_bench_mini],
    # dataset=["/mnt/inside_15T/PPG_dataset/aibot/ypj_finetune/datasets/GPT4all/gpt4all.jsonl"],
    dataset=["/mnt/inside_15T/PPG_dataset/aibot/ypj_finetune/datasets/my_data.jsonl"],
    # train_dataset_sample=1000,
    train_dataset_sample=2000,
    logging_steps=5,
    max_length=2048,
    learning_rate=5e-5,
    warmup_ratio=0.4,
    # output_dir='output',
    output_dir='/mnt/inside_15T/PPG_dataset/aibot/ypj_model/out',
    lora_target_modules=['ALL'],
    self_cognition_sample=500,
    # self_cognition_sample=1000,
    model_name=['反诈模型', 'Anti-Fraud'],
    model_author=['培杰', 'BenSmith'],
    sft_type="lora",
    # rank(矩阵的秩)和alpha增加需要一起增加
    # lora_rank=8,
    # lora_alpha=32,
    lora_rank=256,
    # lora_alpha=512,
    lora_alpha=1024,
    lora_bias_trainable="none",
    seed=42,
    # seed=20,
    # seed=50,
    # dtype='AUTO',
    dtype='fp32',
    # galore_rank=128,
    galore_rank=256,
    # galore_update_proj_gap=50,
    galore_update_proj_gap=25,
    lora_lr_ratio=16.0,
    # lora_lr_ratio=32.0,
    # lora_lr_ratio=4.0,
    # num_train_epochs=1)
    # num_train_epochs=2)
    num_train_epochs=10)
output = sft_main(sft_args)
best_model_checkpoint = output['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')