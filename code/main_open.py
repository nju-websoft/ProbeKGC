import os
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
    GenerationConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
)

from knowledge_graph import KnowledgeGraph
from prompt_hub import Prompt
from chat_api import call_chat_api

# set your model path here
MODEL_PATH = {
    "llama3.1-8b": "Llama-3.1-8B-Instruct",
    "llama3.1-70b": "Llama-3.1-70B-Instruct",
    "qwen2.5-7b": "Qwen2.5-7B-Instruct",
    "qwen2.5-72b": "Qwen2.5-72B-Instruct",
}


def generate(prompts: List, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, use_context, fixed_option):
    option_idxs = tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"])
    logits_list = []
    for prompt in tqdm(prompts):
        system_prompt, user_prompt = prompt.question_prompt(use_cot=False, return_one=False, use_context=use_context, fixed_option=fixed_option)
        msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        
        input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()  
        attention_mask = input_ids.ne(tokenizer.pad_token_id) 
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            generation_config=GenerationConfig(
                max_new_tokens=16,
                min_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                output_logits=True, return_dict_in_generate=True,
            ),
         )

        logits = output.logits[0][0, option_idxs]
        logits_list.append(logits)
        
    logits = torch.stack(logits_list, dim=0).cpu()
    return logits


def main(
    dataset: str = "",  # select from {ProbeKGC-FB|ProbeKGC-WN|ProbeKGC-YG}
    model_name: str = "",  # select from {llama3.1-8b|llama3.1-70b|qwen2.5-7b|qwen2.5-72b}, this also supports other LLMs
    run_name: str = "",  # set a folder to save the output files
    data_mode: str = None,  # select from {simple|medium|hard}
    fixed_option_id: str = None,  # set this to {A|B|C|D} to for Bias Evaluation
    use_context: bool = False, # set this to {True} to enable RAG, used for Knowledge Enhancement Evaluation
    seed: int = 42,
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

    if os.path.exists(os.path.join(output_dir, f"logits_{data_mode}.pt")):
        print("Logits already exists, skip")
        return
    
    print(f"Loading {data_mode} data")
    data = json.load(open(os.path.join(data_dir, f"test_{data_mode}.json"), "r", encoding="utf-8"))
    prompts = [Prompt(item, kg) for item in data]

    logits = generate(prompts, tokenizer, model, fixed_option=fixed_option_id, use_context=use_context)
    torch.save(logits, os.path.join(output_dir, f"logits_{data_mode}.pt"))

if __name__ == "__main__":
    Fire(main)