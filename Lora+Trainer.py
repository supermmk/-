from transformers import Trainer, DataCollatorForSeq2Seq


# 1. 自己处理数据（拼成 text，并 Tokenize）
def preprocess_function(examples):
    inputs = []
    targets = []
    for i in range(len(examples['instruction'])):
        # 拼 prompt
        prompt = f"### Instruction:\n{examples['instruction'][i]}\n\n"
        if examples['input'][i]:
            prompt += f"### Input:\n{examples['input'][i]}\n\n"
        prompt += f"### Response:\n"

        answer = examples['output'][i] + tokenizer.eos_token

        inputs.append(prompt)
        targets.append(answer)

    # Tokenize 输入和回答
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True)

    # 核心：把 prompt 部分的 labels 设为 -100（不计算 loss）
    # 这里简单演示，具体需要根据 padding 对齐长度
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels["input_ids"]
    }


# 2. 应用到数据集
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# 3. 自己弄个 Collator（负责补全和生成标签）
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

# 4. 手动包装 LoRA（不用 SFTTrainer 的自动包装）
from peft import get_peft_model

model = get_peft_model(model, peft_config)

# 5. 用最原生的 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()