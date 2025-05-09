import os
# CUDA_DEVICE_ORDER=PCI_BUS_ID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# TOKENIZERS_PARALLELISM=(true | false)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import json
import random
import inspect
from fire import Fire
from typing import List, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import (
    set_seed,
    BitsAndBytesConfig,
    GenerationConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
)

from knowledge_graph import KnowledgeGraph
from prompt_hub import Prompt
from utils import score, get_logger, print_msg
from chat_api import call_chat_api

MODEL_PATH = {
    "llama3.1-8b": "/data1/shares/Llama-3.1-8B-Instruct",
    "llama3.1-70b": "/data1/shares/Llama-3.1-70B-Instruct",
    "qwen2.5-7b": "/data1/shares/Qwen2.5-7B-Instruct",
    "qwen2.5-72b": "/data1/shares/Qwen2.5-72B-Instruct",
    "deepseek-r1": "/data1/shares/DeepSeek-R1-Distill-Llama-8B", # 不适合获取概率, 因为会先思考
}

def generate(prompts: List, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, use_context, use_cot):
    msgs = []
    for prompt in tqdm(prompts):
        system_prompt, user_prompt = prompt.question_prompt(use_cot=use_cot, return_one=False, use_context=use_context, fixed_option=None)
        msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()  # (1, seq_len)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)  # (1, seq_len)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            generation_config=GenerationConfig(
                max_new_tokens=1024,
                min_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                output_logits=True, return_dict_in_generate=True,
            ),
         )
        # output: sequences: Tensor(1, new_seq_len), logits: Tuple[Tensor]

        output = tokenizer.decode(output.sequences[0, input_ids.shape[-1]:], skip_special_tokens=False).strip()
        msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}, {"role": "assistant", "content": output}]
        msgs.append(msg)
        print_msg(msg)
    return msgs
    

def main(
    dataset: str = "",  # select from {ProbeKGC-FB|ProbeKGC-WN|ProbeKGC-YG}
    model_name: str = "", # select from {llama3.1-8b|llama3.1-70b|qwen2.5-7b|qwen2.5-72b}, this also supports other LLMs
    run_name: str = "",  # set a folder to save the output files
    data_mode: str = None,  # select from {simple|medium|hard}
    use_context: bool = False,  # set this to {True} to enable RAG, used for Knowledge Enhancement Evaluation
    use_cot: bool = False,  # set this to {True} to enable CoT, used for Knowledge Enhancement Evaluation
    seed: int = 42,  # random seed
):
    assert dataset in ["ProbeFB", "ProbeWN", "ProbeYG"], f"Not supported dataset {dataset}"
    set_seed(seed)

    if model_name == "qwen2.5-72b" or model_name == "llama3.1-70b":
        quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    else:
        quantization_config = None
    model_path = MODEL_PATH[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )

    data_dir = os.path.join("dataset", dataset)
    output_dir = os.path.join("output", dataset, run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Set data dir to {data_dir}")
    print(f"Set output dir to {output_dir}")

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    args_dict = {arg: values[arg] for arg in args}
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    args_dict = {arg: values[arg] for arg in args}
    if not os.path.exists(os.path.join(output_dir, "log.txt")):
        with open(os.path.join(output_dir, "log.txt"), "w", encoding="utf-8") as fout:
            fout.write(json.dumps(args_dict, indent=4, ensure_ascii=False))
    
    print("Loading Knowledge Graph")
    kg = KnowledgeGraph(data_dir)

    print(f"Loading {data_mode} data")
    data = json.load(open(os.path.join(data_dir, f"test_{data_mode}.json"), "r", encoding="utf-8"))
    prompts = [Prompt(item, kg) for item in data]

    # for data_mode in ["simple", "medium", "hard"]:
    output_path = os.path.join(output_dir, f"msg_{data_mode}.jsonl")
    if os.path.exists(output_path):
        print(output_path, "exitst, skip")
        return

    msgs = generate(prompts, tokenizer, model, use_cot=use_cot, use_context=use_context)
    with open(output_path, "a", encoding="utf-8") as fout:
        for msg in msgs:
            fout.write(json.dumps(msg, ensure_ascii=False) + "\n")

    
if __name__ == "__main__":
    Fire(main)
    