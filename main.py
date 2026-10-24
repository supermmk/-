from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_from_disk
from trl import SFTConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model="./Qwen2.5-0.5B",
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)

tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-0.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_from_disk("./data/alpaca_sft_clean")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_args = SFTConfig(
    output_dir="./qwen-lora-sft",
    max_length=512,
    dataset_text_field="texts",

    num_train_epochs=3,

    learning_rate=2e-4,
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_steps=50,
    max_grad_norm=1.0,

    save_strategy="epoch",
    save_total_limit=2,
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",

    logging_strategy="epoch",

    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
)