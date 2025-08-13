import pandas as pd
import numpy as np
import os
from typing import List
import requests
from openai import OpenAI
from huggingface_hub import InferenceClient


def llama_answer(
    question: str,
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    external_knowledge=None,
) -> str:

    prompt = "You are an expert answering common sense question.\n"
    if external_knowledge:
        prompt += f"You may find this information useful: {external_knowledge}.\n"
    prompt += f"Question: {question}\n"
    client = InferenceClient(api_key="HF_API_TOKEN")

    messages = [{"role": "user", "content": prompt}]

    stream = client.chat.completions.create(
        model=model_name, messages=messages, max_tokens=5000, stream=True
    )
    # Initialize an empty string to store the accumulated result
    result_string = ""

    # Iterate through the stream and accumulate content into the result string
    for chunk in stream:
        result_string += chunk.choices[0].delta.content
    return result_string


def gpt_answer(question: str, model_name="gpt-4o-mini", external_knowledge=None) -> str:
    client = OpenAI()
    prompt = f"You are an expert at common sense.\n"
    if external_knowledge:
        prompt += f"You may find this information useful: {external_knowledge}\n"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]
    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        output = response.choices[0].message.content.strip()
        return output
    except Exception as e:
        print(f"Error: {e}")
        return None
