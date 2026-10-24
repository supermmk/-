import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# ============================================================
# 1. 基础设置
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# ============================================================
# 2. 自己处理数据（拼文本 + Tokenize + Mask 掩码）
# ============================================================
dataset = load_dataset("parquet", data_files="./alpaca_data/data/*.parquet", split="train")
# 取 1000 条测试，跑通再说
dataset = dataset.select(range(1000))


def preprocess_function(examples):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for i in range(len(examples['instruction'])):
        # 1. 拼装完整的 Prompt
        prompt = f"### Instruction:\n{examples['instruction'][i]}\n\n"
        if examples['input'][i]:
            prompt += f"### Input:\n{examples['input'][i]}\n\n"
        prompt += f"### Response:\n"

        answer = examples['output'][i] + tokenizer.eos_token

        # 2. 分别 Tokenize 提问和回答
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

        # 3. 拼接在一起
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids  # 👈 核心：Prompt 部分的 Label 变 -100

        # 4. 截断（防止超长）
        max_len = 512
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append([1] * len(input_ids))

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }


# 应用预处理
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# ============================================================
# 3. 配置 LoRA（手动包装）
# ============================================================
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
# 必须自己包！
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 打印看看是不是只有 1% 的参数在训练

# ============================================================
# 4. 配置 DataCollator
# ============================================================
# 用 Seq2Seq 的 Collator，它不仅能补齐 input_ids，还能用 -100 补齐 labels！
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100  # 保证 padding 的 label 也是 -100
)

# ============================================================
# 5. 原生 TrainingArguments（不需要 SFTConfig 了）
# ============================================================
training_args = TrainingArguments(
    output_dir="./qwen-native-sft",
    num_train_epochs=1,
    per_device_train_batch_size=4,  # 因为我们自己写了 Mask，可以用稍微大一点的 batch
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)

# ============================================================
# 6. 组装原生 Trainer
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,  # 传入我们自己配置的 Collator
)

# ============================================================
# 7. 开始训练
# ============================================================
print("开始原生 SFT 训练...")
trainer.train()

# 8. 保存
model.save_pretrained("./qwen-native-lora")
tokenizer.save_pretrained("./qwen-native-lora")