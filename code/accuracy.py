import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import numpy as np
from fire import Fire
from typing import List, Dict
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from transformers import set_seed

from knowledge_graph import KnowledgeGraph


OPTION_IDS = ["A", "B", "C", "D"]
KG = None


def get_acc_list_from_log_probs(data: List[Dict], log_probs: torch.Tensor, fixed_option=None):
    idx2option = {idx: ID for idx, ID in enumerate(OPTION_IDS)}
    sorted_idxs = torch.argsort(log_probs, dim=-1, descending=True)
    options = [idx2option[x] for x in sorted_idxs[:, 0].cpu().numpy().tolist()]
    
    acc_list = []
    for item, ans_id in zip(data, options):
        h, r, t = item["triple"]
        task = item["task"]
        if task == "head prediction":
            gold = h
        elif task =="tail prediction":
            gold = t
        else:
            assert 0

        if fixed_option is None: 
            if item[ans_id] == gold:
                acc_list.append(1)
            else:
                acc_list.append(0)
        else:
            if ans_id == fixed_option:
                acc_list.append(1)
            else:
                acc_list.append(0)
    return acc_list

def get_log_probs_from_jsonl(file_path: str):
    log_probs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            _log_probs = []
            hits = 0
            for ID in OPTION_IDS:
                if ID in item:
                    _log_probs.append(item[ID])
                    hits += 1
                else:
                    _log_probs.append(-1e5)
            log_probs.append(_log_probs)
            if hits == 0:
                print("None of options in:", item)
    return torch.tensor(log_probs).cuda()

def get_log_probs_from_pt(file_path: str, temperature=1.):
    logits = torch.load(file_path, map_location="cuda:0", weights_only=True)
    log_probs = F.log_softmax(logits/temperature, dim=-1)
    return log_probs


def main(
    dataset: str = "",  # select from {ProbeKGC-FB|ProbeKGC-WN|ProbeKGC-YG}
    run_name: str = "",  # set a folder to save the output files
    data_mode: str = None,  # select from {simple|medium|hard}
    fixed_option: str = None,  # set this to {A|B|C|D} to for Bias Evaluation
    seed: int = 42,
):
    assert dataset in ["ProbeKGC-FB", "ProbeKGC-WN", "ProbeKGC-YG"], f"Not supported dataset {dataset}"
    set_seed(seed)

    data_dir = os.path.join("dataset/ProbeKGC", dataset)
    output_dir = os.path.join("output", dataset, run_name)
    assert os.path.exists(output_dir), f"Output dir {output_dir} not exists"
    print(f"Set data dir to {data_dir}")
    print(f"Set output dir to {output_dir}")
        
    data = json.load(open(os.path.join(data_dir, f"test_{data_mode}.json"), "r", encoding="utf-8"))
    print(f"data num: {len(data)}")

    if "gpt" in run_name:
        log_probs = get_log_probs_from_jsonl(os.path.join(output_dir, f"logits_{data_mode}.jsonl"))
    elif "llama" in run_name or "qwen" in run_name:
        log_probs = get_log_probs_from_pt(os.path.join(output_dir, f"logits_{data_mode}.pt"))
    else:
        assert 0, run_name
    
    loss_fn = torch.nn.KLDivLoss(reduction="none")
    target_probs = torch.tensor([.25, .25, .25, .25]).to("cuda:0")
    score_dict = dict()
    acc_dict = dict()
    
    acc_list = get_acc_list_from_log_probs(data, log_probs, fixed_option=fixed_option)
    acc_dict[data_mode] = acc_list
    print(f"{data_mode} acc ({len(acc_list)}): {np.mean(np.array(acc_list)):.4f}")

    kl_div = torch.mean(loss_fn(log_probs, target_probs), dim=-1)
    scores = 1. - torch.exp(-kl_div)
    score_dict[data_mode] = scores
    print(f"{data_mode} nkd ({scores.shape}): {torch.mean(scores):.4f}")


if __name__ == "__main__":
    Fire(main)
           

