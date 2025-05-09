import os
import json
from tqdm import tqdm
from fire import Fire
from argparse import ArgumentParser
from transformers import set_seed
from knowledge_graph import KnowledgeGraph
from prompt_hub import Prompt
from chat_api import call_chat_api



def main(
    dataset: str = "",  # select from {ProbeKGC-FB|ProbeKGC-WN|ProbeKGC-YG}
    run_name: str = "",  # set a folder to save the output files
    data_mode: str = "",  # select from {simple|medium|hard}
    api_model: str = "",  # select from {gpt-3.5-turbo|gpt-4o-mini}, this also supports other APIs
    use_cot: bool = False,  # set this to {True} to enable Cot, used for Knowledge Enhancement Evaluation
    use_context: bool = False,  # set this to {True} to enable RAG, used for Knowledge Enhancement Evaluation
    fixed_option_id: str = None,  # set this to {A|B|C|D} to for Bias Evaluation
    logprobs: bool = True,  # set this to {True} to ensure that APIs return the logprobs of top-{top_logprobs} tokens, set this to {False} when use_cot is {True}
    top_logprobs: int = 20,   # the number of tokens which need to return the logprobs
    seed: int = 42, 
):
    assert dataset in ["ProbeKGC-FB", "ProbeKGC-WN", "ProbeKGC-YG"], f"Not supported dataset {dataset}"
    set_seed(seed)

    data_dir = os.path.join("dataset/ProbeKGC", dataset)
    output_dir = os.path.join("output", dataset, run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = json.load(open(os.path.join(data_dir, f"test_{data_mode}.json"), "r", encoding="utf-8"))

    kg = KnowledgeGraph(data_dir)
    prompts = [Prompt(item, kg) for item in data]

    msg_path = os.path.join(output_dir, f"msg_{data_mode}.jsonl")
    logits_path = os.path.join(output_dir, f"logits_{data_mode}.jsonl")
    if not os.path.exists(msg_path):
        offset = 0
    else:
        with open(msg_path, "r", encoding="utf-8") as fin:
            offset = len(fin.readlines())

    for idx in tqdm(range(offset, len(prompts), 1), total=len(prompts), desc="Call API", initial=offset):
        prompt = prompts[idx]
        question = prompt.question_prompt(
            return_one=True, use_cot=use_cot, use_context=use_context, fixed_option=fixed_option_id
        )
        msg = [{"role": "user", "content": question}]

        response, token2prob = call_chat_api(msg, model=api_model, max_tokens=1024, logprobs=logprobs, top_logprobs=top_logprobs)
        msg.append({"role": "assistant", "content": response})

        with open(msg_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        if token2prob is not None:
            with open(logits_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(token2prob, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    Fire(main)
