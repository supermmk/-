import torch
from datasets import load_dataset, DatasetDict

dataset = load_dataset(
    "parquet",
    data_files="./alpaca_data/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
    split="train",
)

def formatting_prompts(examples):
    texts = []
    for i in range(len(examples['instruction'])):
        text = f"### Instruction:\n{examples['instruction'][i]}\n\n"
        if examples['input'][i]:
            text += f"### Input:\n{examples['input'][i]}\n\n"
        text += f"### Output:\n{examples['output'][i]}\n\n"
        texts.append(text)
    return {"texts": texts}

clean_dataset = dataset.map(formatting_prompts, batched=True, remove_columns=dataset.column_names)

split_dataset = clean_dataset.train_test_split(test_size=0.05, seed=42)
final_dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"],
})

save_path = "./data/alpaca_sft_clean"
final_dataset.save_to_disk(save_path)

print(f"训练集大小: {len(final_dataset['train'])} 条")
print(f"验证集大小: {len(final_dataset['validation'])} 条")
print("保存成功！")