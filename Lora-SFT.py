from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer

# 1. 加载模型和 Tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# 2. 准备 SFT 数据集（这里以 Alpaca 格式为例）
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 3. 定义数据格式化函数（把 instruction/input/output 拼成一句话）
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Instruction:\n{example['instruction'][i]}\n\n### Response:\n{example['output'][i]}"
        output_texts.append(text)
    return output_texts

# 4. 配置 LoRA（SFT 标配）
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Qwen 的注意力层
    task_type="CAUSAL_LM"
)

# 5. 组装 SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func, # 这里自动处理数据
    max_seq_length=512,                      # 截断长度
    args=...,                                # 这里填 TrainingArguments
)

# 6. 开跑
trainer.train()