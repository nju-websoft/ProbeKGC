import os
import re
import json
from fire import Fire
from typing import List, Dict
from collections import defaultdict, Counter
from transformers import set_seed
from utils import load_msg

"""
compute the accuracy for each sample
"""


OPTION_IDS = ["A", "B", "C", "D"]

def get_acc_list_normal(data: List[Dict], msgs: List[Dict]):
    """
    return a List consists of {-1, 0, 1}, in which 1 means correct, 0 means wrong, -1 means not found
    gpt-3.5-turbo may return "A" or "A. entity name"
    """
    pattern = r"^([A-D])\.\s"

    acc_list = []
    for item, msg in zip(data, msgs):
        h, r, t = item["triple"]
        task = item["task"]
        if task == "head prediction":
            gold = h
        elif task =="tail prediction":
            gold = t
        else:
            assert 0
        
        response = msg[-1]["content"]
        if response in OPTION_IDS + [ID+"." for ID in OPTION_IDS]:
            if item[response[0]] == gold:
                acc_list.append(1)
            else:
                acc_list.append(0)
            continue

        matches = re.findall(pattern, response)
        if len(matches) != 1:
            print("="*40)
            print("response:", response)
            print("matches:", matches)
            print("="*40)
            acc_list.append(-1)
            continue
        if item[matches[0]] == gold:
            acc_list.append(1)
        else:
            acc_list.append(0)

    return acc_list

def get_acc_list_cot(data: List[Dict], msgs: List[Dict]):
    """
    return a List consists of {-1, 0, 1}, in which 1 means correct, 0 means wrong, -1 means not found
    """
    # pattern = r"\*\*The answer is ([A-D])\.\*\*"
    pattern = r"[T|t]he answer is ([A-D])\."

    acc = []
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

        if len(matches) ==2 and matches[0] == matches[1]:
            matches = [matches[0]]

        if len(matches) != 1:
            print(response)
            print(matches)
            print("="*80)
            acc.append(-1)
            continue

        if item[matches[0]] == gold:
            acc.append(1)
        else:
            acc.append(0)

    return acc


def head_tail_consistency(acc_list):
    num = len(acc_list)
    assert num % 2 == 0, num

    true_cnt, false_cnt, void_cnt = 0, 0, 0
    for x in acc_list:
        if x == 1:
            true_cnt += 1
        if x == 0:
            false_cnt += 1
        if x == -1:
            void_cnt += 1
    
    both_true, both_false, both_void = 0, 0, 0
    for idx in range(0, num, 2):
        if acc_list[idx] == 1 and acc_list[idx+1] == 1:
            both_true += 1
        if acc_list[idx] == 0 and acc_list[idx+1] == 0:
            both_false += 1
        if acc_list[idx] == -1 and acc_list[idx+1] == -1:
            both_void += 1
    print(f"both_true: {both_true*2/true_cnt:.4f}, both_false: {both_false*2/false_cnt:.4f}, both_void: {both_void*2/void_cnt:.4f}")

def compute_difference(data_dict: Dict, acc_dict: Dict):
    acc_idxs = {mode: {idx for idx, x in enumerate(acc_dict[mode]) if x == 1} for mode in acc_dict}
    
    simple_only = acc_idxs["simple"] - acc_idxs["medium"] - acc_idxs["hard"]
    medium_only = acc_idxs["medium"] & acc_idxs["simple"] - acc_idxs["hard"]
    hard_only = acc_idxs["hard"] & acc_idxs["simple"] & acc_idxs["medium"]

    print(len(simple_only), len(medium_only), len(hard_only))

    simple_counter = Counter([data_dict["simple"][idx]["triple"][1] for idx in simple_only])
    medium_counter = Counter([data_dict["medium"][idx]["triple"][1] for idx in medium_only])
    hard_counter = Counter([data_dict["hard"][idx]["triple"][1] for idx in hard_only])

    # print(simple_counter)
    # print(medium_counter)
    # print(hard_counter)
            

def main(
    dataset: str,  # select from {ProbeKGC-FB|ProbeKGC-WN|ProbeKGC-YG}
    run_name: str,  # set a folder to save the output files
    data_mode: str,  # select from {simple|medium|hard}
    seed: int = 42,  # random seed
):
    assert dataset in ["ProbeKGC-FB", "ProbeKGC-WN", "ProbeKGC-YG"], f"Not supported dataset {dataset}"
    set_seed(seed)

    data_dir = os.path.join("dataset/ProbeKGC", dataset)
    output_dir = os.path.join("output", dataset, run_name)
    assert os.path.exists(output_dir), f"Output dir {output_dir} not exists"
    print(f"Set data dir to {data_dir}")
    print(f"Set output dir to {output_dir}")

    data = json.load(open(os.path.join(data_dir, f"test_{data_mode}_sample.json"), "r", encoding="utf-8"))
    msgs = load_msg(f"{output_dir}/{dataset}/{run_name}/msg_{data_mode}.jsonl")

    if run_name.endswith("cot"):
        get_acc_list = get_acc_list_cot
    else:
        get_acc_list = get_acc_list_normal

    acc_list = get_acc_list(data, msgs)

    acc_cnt, valid_cnt, void_cnt = 0, 0, 0
    for idx, x in enumerate(acc_list):
        if x == 1:
            acc_cnt += 1
        if x != -1:
            valid_cnt += 1
        if x == -1:
            void_cnt += 1
    assert valid_cnt + void_cnt == len(acc_list)

    print(f"{data_mode} accuracy: {acc_cnt} / {valid_cnt} = {acc_cnt/(valid_cnt):.4f}")
    print(f"{data_mode} out of scope count: {void_cnt}")
    print(f"data path is: {data_dir}/test_{data_mode}.json")
    print(f"msg path is: output/{dataset}/{run_name}/msg_{data_mode}.jsonl")


if __name__ == "__main__":
    Fire(main)
