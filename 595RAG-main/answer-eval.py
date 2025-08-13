import os
from openai import OpenAI
import requests
import numpy as np

client = OpenAI()

API_URL = (
    "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
)
headers = {"Authorization": f"Bearer YOUR_HF_API_KEY"}


def check_answer_equivalence(question, answer1, answer2, model="gpt-4o-mini"):
    """
    Determines whether two answers are equivalent based on the output of a GPT model.

    :param question: The question being evaluated.
    :param answer1: The first answer.
    :param answer2: The second answer.
    :param model: The GPT model to use, default is gpt-4o-mini.
    :return: True if the answers are equivalent, otherwise False.
    """
    prompt = f"""
    Are the following two answers to the given question equivalent? Do not consider whether the answers are right or wrong, but only whether they are equivalent. Directly state \u201dYes\u201d or \u201dNo\u201d.

    Question: {question}
    Answer 1: {answer1}
    Answer 2: {answer2}

    Are the two answers equivalent?"""
    messages = [
        {"role": "system", "content": f"You are an expert at common sense."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(model=model, messages=messages)
        output = response.choices[0].message.content.strip()
        if output.lower() == "yes":
            return True
        elif output.lower() == "no":
            return False
        else:
            raise ValueError(f"Unexpected model output: {output}")
    except Exception as e:
        print(f"Error: {e}")
        return None


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Hugging Face API error: {response.status_code} - {response.text}"
        )


def calculate_similarity(answer1, answer2):

    payload = {"inputs": {"source_sentence": answer1, "sentences": [answer2]}}

    try:
        output = query(payload)
        similarity_score = output[0]
        return similarity_score
    except Exception as e:
        print(f"Error: {e}")
        return None
