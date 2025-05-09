import json
import requests
from openai import OpenAI

from typing import AnyStr, List

MAX_TRY = 10

GPT_url = "https://api.openai.com/v1"
GPT_key = ""


client = None


def call_chat_api(
        message: List, 
        model="gpt-4o-mini", 
        max_tokens=1024, 
        temperature=0., 
        presence_penalty=0., 
        frequency_penalty=0., 
        top_p=1.,
        logprobs=False,
        top_logprobs=None,
    ) -> AnyStr:
    
    global client
    if client is None:
        client = OpenAI(api_key=GPT_key, base_url=GPT_url)

    response = client.chat.completions.create(
        model=model,
        messages=message,
        stream=False,
        max_tokens=max_tokens, 
        temperature=temperature, 
        top_p=top_p,
        presence_penalty=presence_penalty, 
        frequency_penalty=frequency_penalty, 
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )
    response_text = response.choices[0].message.content
    if logprobs:
        topk = response.choices[0].logprobs.content[0].top_logprobs
        topk_dict = {item.token: item.logprob for item in topk}

    if not logprobs:
        return response_text, None
    else:
        return response_text, topk_dict
