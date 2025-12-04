这里演示的是从本地加载kernel使用的教程

```
import time
import logging
from pathlib import Path
from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer

from kernels import (
    Device,
    LayerRepository,
    LocalLayerRepository,
    Mode,
    register_kernel_mapping,
)


# Set the level to `DEBUG` to see which kernels are being called.
logging.basicConfig(level=logging.DEBUG)

model_name = "/root/Qwen3-0.6B"

_KERNELS_MAPPING: dict[str, dict[Union[Device, str], LocalLayerRepository]] = {
    "RMSNorm": {
        "npu": LocalLayerRepository(
            repo_path=Path("/root/rmsnorm/"),
            package_name="rmsnorm",
            layer_name="rmsnorm",
        )
    }
}

# inherit_mapping=False,
register_kernel_mapping(_KERNELS_MAPPING)

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    use_kernels=True,
)

# Prepare the model input
prompt = "Output the first 20 digits of pi."
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
