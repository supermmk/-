# ============================================================
# SFT + LoRA 微调 Qwen-0.5B 完整流程
# ============================================================

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ============================================================
# 1. 设备检测与模型加载
# ============================================================
# 检测 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载模型，并直接搬运到 GPU
model = AutoModelForCausalLM.from_pretrained(
    "./Qwen2.5-0.5B",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B")

# 💡 很多模型没有 pad_token，SFT 需要它，通常用 eos_token 替代
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 2. 准备 SFT 数据集（直接从 Parquet 读取）
# ============================================================
# 加载本地下载好的 Parquet 文件，并直接用 split="train" 取出训练集
dataset = load_dataset(
    "parquet",
    data_files="./alpaca_data/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
    split="train"
)
# 打印一下确认数据量
print(f"训练集条数: {len(dataset)}")


# ============================================================
# 3. 定义数据格式化函数
# ============================================================
def formatting_prompts_func(example):
    """
    把 Alpaca 格式的表格拼成一段完整的文本。
    """
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Instruction:\n{example['instruction'][i]}\n\n"
        # 如果有额外的 input，就拼进去
        if example['input'][i]:
            text += f"### Input:\n{example['input'][i]}\n\n"
        text += f"### Response:\n{example['output'][i]}"
        output_texts.append(text)
    return {"texts": output_texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
# ============================================================
# 4. 配置 LoRA
# ============================================================
peft_config = LoraConfig(
    r=16,  # 秩：控制新增参数矩阵的大小（8~64）
    lora_alpha=32,  # 缩放系数：通常设为 r 的 2 倍
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 插在 Qwen 的注意力层
    lora_dropout=0.05,  # Dropout 概率：防过拟合
    bias="none",  # 不训练偏置项
    task_type="CAUSAL_LM"  # 任务类型：因果语言模型
)

# ============================================================
# 5. 配置训练参数（分好类，防爆显存）
# ============================================================
sft_args = SFTConfig(
    # ---- 1. 基础配置 ----
    output_dir="./qwen-lora-sft",  # 模型和 checkpoint 的保存路径
    max_length=512, # 截断长度（超过 512 的文本会被切掉）
    dataset_text_field="texts",  # 👈 关键：告诉 SFTTrainer 哪一列是拼好的文本

    # ---- 2. 训练时长与批次 ----
    num_train_epochs=3,  # 训练几轮
    per_device_train_batch_size=2,  # 每张卡训练的 batch 大小
    gradient_accumulation_steps=8,  # 梯度累积（等效 batch = 2 * 8 = 16）

    # ---- 3. 优化器与学习率 ----
    learning_rate=2e-4,  # 学习率（LoRA 通常比全量微调大）
    weight_decay=0.01,  # 权重衰减
    optim="paged_adamw_8bit",  # 优化器（推荐 8bit 防爆显存）
    lr_scheduler_type="cosine",  # 余弦调度
    warmup_steps=50,  # 预热比例（前 3% 的步数慢慢提升学习率）
    max_grad_norm=1.0,  # 梯度裁剪（防止梯度爆炸）

    # ---- 4. 评估与保存 ----
    # eval_strategy="epoch",           # 评估策略：每个 epoch 评一次
    # load_best_model_at_end=True,     # 训练结束加载最好的模型
    # metric_for_best_model="eval_loss",# 依据什么指标选最好
    save_strategy="epoch",  # 保存策略：每个 epoch 存一次
    save_total_limit=2,  # 只保留最近 2 个 checkpoint（防爆硬盘）

    # ---- 5. 日志记录 ----
    logging_steps=10,  # 每 10 步打印一次日志

    # ---- 6. 显存与精度 ----
    fp16=torch.cuda.is_available(),  # 有 CUDA 才开半精度
    gradient_checkpointing=True,  # 梯度检查点（用计算时间换显存，小显存必开）
)

# ============================================================
# 6. 组装 SFTTrainer
# ============================================================
trainer = SFTTrainer(
    model=model,  # 基础模型
    train_dataset=dataset,  # 训练数据集
    peft_config=peft_config,  # LoRA 配置（自动冻结底座，挂载 LoRA）
    args=sft_args,  # 训练指挥官
)

# ============================================================
# 7. 开跑！
# ============================================================
print("开始训练...")
trainer.train()

# ============================================================
# 8. 保存 LoRA 权重
# ============================================================
# 保存的只有 LoRA 适配器权重（通常只有几十 MB），不是整个模型
trainer.model.save_pretrained("./qwen-lora-final")
tokenizer.save_pretrained("./qwen-lora-final")
print("训练完成，LoRA 权重已保存至 ./qwen-lora-final")