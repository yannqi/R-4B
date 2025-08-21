# R-4B: Incentivizing General-Purpose Auto-Thinking in MLLMs via Bi-Mode Integration

[[📚 Arxiv Paper (Coming soon)](https://huggingface.co/YannQi/R-4B)] [[🤗 Hugging Face](https://huggingface.co/YannQi/R-4B)]  [[🤖️ ModelScope](https://huggingface.co/YannQi/R-4B)] [[💻 Code](https://github.com/yannqi/R-4B)]

<div align="center">
<img src="asset/logo_R_4B.png" alt="logo" width="38" /> 
</div>

<div align="center">
  <img src="asset/R-4B.png" width="100%" alt="R-4B Performance">
</div>

## ⭐️ Introduction

In this repo, we present **R-4B**, a multimodal large language model designed for general-purpose auto-thinking, autonomously switching between step-by-step thinking and direct response generation based on task complexity. This capability enables R-4B to deliver high-quality responses while significantly improving inference efficiency and reducing computational costs.

The development of R-4B follows a two-stage training paradigm:
(1) Bi-mode Annealing, which establishes both thinking and non-thinking capabilities for VQA; and
(2) Bi-mode Policy Optimization (BPO), which enables the model to adaptively switch between thinking and non-thinking modes based on input demands.

R-4B achieves state-of-the-art performance among models of its scale. In evaluations across multiple public benchmarks, R-4B outperforms Qwen2.5-VL-7B on nearly all tasks. Notably, it matches or exceeds the performance of the much larger Kimi-VL-Thinking-2506 (3B activated, 16B total parameters).

## 🔥 Quickstart

Below, we provide simple examples to show how to use R-4B with 🤗 Transformers.

### Using 🤗 Transformers to Chat

> [!NOTE]
> Following Qwen3, we also offer a hard switch mechanism that lets users dynamically control the model's behavior.

```python
import requests
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image

model_path = "YannQi/R-4B"

model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to('cuda')

# default processer
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"{image_file}",
            },
            {"type": "text", "text": "描述该图片。"},
        ],
    }
]

# Preparation for inference

text_auto_thinking = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)  #  thinking_mode='long' for thinking mode;  thinking_mode='short' for non-thinking mode; Defalut is auto-thinking mode.

raw_image = Image.open(requests.get(image_file, stream=True).raw)

inputs_auto_thinking = processor(images=raw_image, text=text_auto_thinking, return_tensors='pt').to(0, torch.float16)

inputs_auto_thinking = inputs_auto_thinking.to("cuda")


# Inference: Generation of the output


generated_ids_auto_thinking = model.generate(**inputs_auto_thinking, max_new_tokens=8192)
generated_ids_trimmed_auto_thinking = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs_auto_thinking.input_ids, generated_ids_auto_thinking)
]


output_text_auto_thinking = processor.batch_decode(
            generated_ids_trimmed_auto_thinking, skip_special_tokens=True, clean_up_tokenization_spaces=False
)


print("Auto Thinking Output:", output_text_auto_thinking)

```

</details>

### Using vLLM for fast R-4B deployment and inference.

- We recommend using vLLM for fast R-4B deployment and inference.

#### Install

The code of R-4B requires the newest vllm now. Please install from local source:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

##### Offline Inference

```python
import os
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
import requests
from io import BytesIO


def load_image(image_path):
    """Load image from URL or local path"""
    if image_path.startswith(('http://', 'https://')):
        response = requests.get(image_path, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
  
    # Convert RGBA to RGB if needed
    if image.mode == "RGBA":
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
  
    return image.convert("RGB")


def main():

    model_path = "YannQi/R-4B/"

    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 5},
        trust_remote_code=True,
        tensor_parallel_size=1,  
        gpu_memory_utilization=0.8, 
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=16384,
    )

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = load_image(image_url)
    text = "Describe this image."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        },
    ]

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    mm_data = {"image": image}
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    print(generated_text)

if __name__ == '__main__':
    main()
```

##### Online Serving

- Serve

```bash
vllm serve \
    yannqi/R-4B \
    --served-model-name rvl \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
```

- Openai Chat Completion Client

```python
import base64
from PIL import Image
from openai import OpenAI


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# image url
image_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "http://images.cocodataset.org/val2017/000000039769.jpg"
                },
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
]

chat_response = client.chat.completions.create(
    model="rvl",
    messages=image_messages,
)
print("Chat response:", chat_response)

# image base64-encoded
image_path = "/path/to/local/image.png"
with open(image_path, "rb") as f:
    encoded_image = base64.b64encode(f.read())
encoded_image_text = encoded_image.decode("utf-8")
image_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image;base64,{encoded_image_text}"
                },
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
]

chat_response = client.chat.completions.create(
    model="rvl",
    messages=image_messages,
)
print("Chat response:", chat_response)
```

## 📈 Experimental Results

<div align="center">
  <img src="asset/performance.png" width="100%" alt="R-4B Performance">
</div>

1. R-4B establishes itself with powerful, state-of-the-art perceptual abilities that are competitive with larger models.
2. In evaluation sets that require complex logical reasoning and mathematical problem-solving, such as WeMath, MathVerse, and LogicVista, R-4B displays a strong performance curve. This highlights its advanced adaptive thinking capacity for logical deduction and solving complex quantitative problems.

## ✒️ Citation

Coming soon!

<!-- If you find our work helpful for your research, please consider citing our work. -->

<!-- 
```bibtex
@misc{R-4B,
      title={R-4B: Adaptive Vision-Language Reasoning for Efficient Inference}, 
      author={Z},
      year={2025},
      eprint={ },
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={ }, 
}
``` -->

## Acknowledgements

R-4B is developed based on the codebases of the following projects: [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT), [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch14-384), [Qwen3](https://github.com/QwenLM/Qwen3), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We sincerely thank these projects for their outstanding work.
