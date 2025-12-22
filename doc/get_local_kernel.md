## 这里是 transformers 从本地加载 kernel 使用的教程

### 前言
近期在 huggingface/transformers 中做了一些工作，使其可以通过 huggingface/kernels 加载本地 kernel 方便 debug 过程。

huggingface/kernels + huggingface/transformers 的相关工作可以看这里：[https://github.com/pt-ecosystem/tracking-map](https://github.com/pt-ecosystem/tracking-map?tab=readme-ov-file#huggingfacekernels-%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E5%B7%A5%E5%85%B7)
### 示例脚本

```python
import time
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig


# Set the level to `DEBUG` to see which kernels are being called.
logging.basicConfig(level=logging.DEBUG)

model_name = "/root/Qwen3"

kernel_mapping = {
    "RMSNorm":
        "/root/liger_kernels:LigerRMSNorm",
}

kernel_config = KernelConfig(kernel_mapping, use_local_kernel=True)

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    kernel_config=kernel_config
)

# Prepare the model input
prompt = "What is the result of 100 + 100?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Warm_up
for _ in range(2):
    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

# Print Runtime
for _ in range(5):
    start_time = time.time()
    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    print("runtime: ", time.time() - start_time)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    print("content:", content)
```
