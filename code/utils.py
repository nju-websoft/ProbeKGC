import os
import re
import json
import random
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from dataclasses import dataclass


def load_msg(msg_path: str):
    msgs = []
    with open(msg_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            msgs.append(json.loads(line.strip()))
    return msgs

def print_msg(msg: List):
    for msg_dict in msg:
        print("*"*30)
        print()
        print(msg_dict["content"])
        print()


def compute_metrics(ranks, top_k=[1, 2, 3, 5, 10]):
    ranks = np.array(ranks)
    metrics = {f"hits@{i}": np.round(np.mean(ranks <= i), 4) for i in top_k}
    metrics["mrr"] = np.round(np.mean(1./ranks), 4)

    return metrics


def score(data: List[Dict], msgs: List[Dict]) -> None:
    assert len(data) == len(msgs), "data and msgs should have the same length"
    pattern = r"\*\*The answer is ([A-Z])\.\*\*"
    
    acc = []
    out_scope_cnt = 0
    for idx, item in enumerate(data):
        h, r, t = item["triple"]
        task = item["task"]
        if task == "head prediction":
            gold = h
        elif task =="tail prediction":
            gold = t
        else:
            assert 0
        
        response = msgs[idx][-1]["content"]
        matches = re.findall(pattern, response)
        if len(matches) != 1:
            out_scope_cnt += 1
            matches = ["A"]

        if item[matches[0]] == gold:
            acc.append(1)
        else:
            acc.append(0)
    print(np.mean(np.array(acc)))
    print(out_scope_cnt)


def get_logger(log_dir: str):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_file = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger