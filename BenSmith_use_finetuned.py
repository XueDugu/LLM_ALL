# Experimental environment: 3090
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
from swift.tuners import Swift

seed_everything(42)

# ckpt_dir = 'output/qwen1half-7b-chat/vx-xxx/checkpoint-xxx'
ckpt_dir = '/mnt/inside_15T/PPG_dataset/aibot/ypj_model/out/qwen1half-0_5b-chat/v1-20240421-110712/checkpoint-93'
model_type = ModelType.qwen1half_0_5b_chat
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 128

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
template = get_template(template_type, tokenizer)

query = '你是什么专家吗？'
response, history = inference(model, template, query)
print(f'response: {response}')
print(f'history: {history}')
"""
[INFO:swift] model.max_model_len: 32768
response: 我是一个基于深度学习的人工智能模型，可以回答各种问题，并提供相关的见解和建议。我会根据您的需求来生成答案或提出建议，包括但不限于科技、商业、艺术等领域的知识。请告诉我您需要什么样的帮助，我将尽力为您提供合适的回答。
history: [['你是什么专家吗？', '我是一个基于深度学习的人工智能模型，可以回答各种问题，并提供相关的见解和建议。我会根据您的需求来生成答案或提出建议，包括但不限于科技、商业、艺术等领域的知识。请告诉我您需要什么样的帮助，我将尽力为您提供合适的回答。']]
"""