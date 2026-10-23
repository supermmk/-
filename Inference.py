import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# 1. 路径配置
# ============================================================
base_model_path = "./Qwen2.5-0.5B"  # 原模型路径
lora_path = "./qwen-lora-final"  # 你训练保存的 LoRA 权重路径

# ============================================================
# 2. 加载模型和 Tokenizer
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2.1 加载原模型
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# 2.2 挂载 LoRA 权重
model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()  # 切换到推理模式


# ============================================================
# 3. 定义对话函数
# ============================================================
def chat(instruction, input_text=""):
    # 拼成和训练时一模一样的格式（⚠️ 必须和 formatting_prompts_func 一致！）
    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += f"### Response:\n"

    # 编码
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # 最多生成 256 个新 token
            temperature=0.7,  # 温度：越低越严谨，越高越发散
            top_p=0.9,  # 核采样
            do_sample=True,  # 采样生成（不贪心）
            repetition_penalty=1.1,  # 重复惩罚，防止复读机
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 只解码新生成的部分（去掉 prompt 部分）
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response


# ============================================================
# 4. 测试一下
# ============================================================
if __name__ == "__main__":
    # 测试 1：普通提问
    print("=" * 50)
    print("测试 1：")
    ans = chat("请用一句话解释什么是机器学习。")
    print(ans)

    # 测试 2：带输入
    print("=" * 50)
    print("测试 2：")
    ans = chat(
        instruction="用 Python 写一个计算阶乘的函数",
        input_text="计算 5 的阶乘"
    )
    print(ans)