import pandas as pd
import numpy as np
import os
from typing import List
import requests
from openai import OpenAI
from huggingface_hub import InferenceClient
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from naiverag import extract_title, embed_text
from get_wikipedia import get_wikipedia_page
from pure_llm import llama_answer, gpt_answer


def feedback_evaluate(
    query: str, retrieved_content: str, model_name="meta-llama/Meta-Llama-3-8B-Instruct"
) -> str:
    """
    Evaluate the relevance of retrieved content for a given query using Llama.

    :param query: The original query.
    :param retrieved_content: The retrieved content to be evaluated.
    :param model_name: The Llama model name.
    :return: Feedback as "Irrelevant", "Partially Relevant", or "Fully Relevant".
    """
    prompt = f"""
    You are an expert evaluator. Your task is to assess how relevant a piece of retrieved content is to a given query.
    The retrieved content should directly address the query and provide useful information.
    
    Here are the evaluation criteria:
    - "Fully Relevant": The content completely addresses the query with accurate and detailed information.
    - "Partially Relevant": The content somewhat addresses the query but lacks important details or includes unrelated information.
    - "Irrelevant": The content does not address the query or is unrelated to it.
    
    Query: {query}
    Retrieved Content: {retrieved_content}
    
    Based on the criteria above, provide one of the following labels: "Fully Relevant", "Partially Relevant", or "Irrelevant".
    """

    client = InferenceClient(api_key="YOUR_API_KEY")

    messages = [{"role": "user", "content": prompt}]

    stream = client.chat.completions.create(
        model=model_name, messages=messages, max_tokens=100, stream=True
    )

    # Initialize an empty string to store the accumulated result
    result_string = ""

    # Iterate through the stream and accumulate content into the result string
    for chunk in stream:
        result_string += chunk.choices[0].delta.content

    # Extract the feedback label from the response
    feedback = result_string.strip()

    # Ensure the feedback is valid
    if feedback not in ["Fully Relevant", "Partially Relevant", "Irrelevant"]:
        feedback = "Irrelevant"  # Default to "Irrelevant" if response is unclear

    return feedback


import pandas as pd
import numpy as np
import os
from typing import List
import requests
from openai import OpenAI
from huggingface_hub import InferenceClient


def feedback_evaluate(
    query: str, retrieved_content: str, model_name="meta-llama/Meta-Llama-3-8B-Instruct"
) -> str:
    """
    Evaluate the relevance of retrieved content for a given query using Llama.

    :param query: The original query.
    :param retrieved_content: The retrieved content to be evaluated.
    :param model_name: The Llama model name.
    :return: Feedback as "Irrelevant", "Partially Relevant", or "Fully Relevant".
    """
    prompt = f"""
    You are an expert evaluator. Your task is to assess how relevant a piece of retrieved content is to a given query.
    The retrieved content should provide useful information. Don't be too strict.

    Here are the evaluation criteria:
    - "Fully Relevant": The content completely addresses the query with accurate and detailed information.
    - "Partially Relevant": The content somewhat addresses the query but lacks important details or includes unrelated information.
    - "Irrelevant": The content does not address the query or is unrelated to it.

    Query: {query}
    Retrieved Content: {retrieved_content}

    Based on the criteria above, return one of the following labels: "Fully Relevant", "Partially Relevant", or "Irrelevant".
    Don't return any other words.
    """

    client = InferenceClient(api_key="YOUR_HF_API_KEY")

    messages = [{"role": "user", "content": prompt}]

    stream = client.chat.completions.create(
        model=model_name, messages=messages, max_tokens=100, stream=True
    )

    # Initialize an empty string to store the accumulated result
    result_string = ""

    # Iterate through the stream and accumulate content into the result string
    for chunk in stream:
        result_string += chunk.choices[0].delta.content

    # Extract the feedback label from the response
    feedback = result_string.strip()
    feedback = feedback.lower()
    print(f"this is feedback:{feedback}")
    if "fully relevant" in feedback:
        return "fully relevant"
    if "partially relevant" in feedback:
        return "partially relevant"
    if "irrelevant" in feedback:
        return "irrelevant"
    return "fully relevant"


def adjust_query(
    query: str, feedback: str, model_name="meta-llama/Meta-Llama-3-8B-Instruct"
) -> str:
    """
    Adjust the query based on feedback using Llama.

    :param query: The original query.
    :param feedback: Feedback from Llama, such as "Irrelevant", "Partially Relevant", or "Fully Relevant".
    :param model_name: The Llama model name.
    :return: Adjusted query.
    """
    if feedback == "Fully Relevant":
        # No adjustment needed if the feedback is "Fully Relevant"
        return query

    prompt = f"""
    You are an expert query refiner. Your task is to improve or adjust a query based on the feedback provided.
    Feedback categories:
    - "Fully Relevant": The query is perfectly fine and no changes are needed.
    - "Partially Relevant": The query is somewhat clear but could be made more specific or include additional details.
    - "Irrelevant": The query needs significant rephrasing to better match the intended context.

    Feedback: {feedback}
    Original Query: {query}

    Based on the feedback, rewrite the query to improve it.
    """

    client = InferenceClient(api_key="YOUR_HF_API_KEY")

    messages = [{"role": "user", "content": prompt}]

    stream = client.chat.completions.create(
        model=model_name, messages=messages, max_tokens=100, stream=True
    )

    # Initialize an empty string to store the accumulated result
    result_string = ""

    # Iterate through the stream and accumulate content into the result string
    for chunk in stream:
        result_string += chunk.choices[0].delta.content

    # Extract the adjusted query from the response
    adjusted_query = result_string.strip()

    # Ensure the adjusted query is not empty; fallback to original query if needed
    if not adjusted_query:
        adjusted_query = query

    return adjusted_query


def iterative_query_improvement(row, max_iterations=1):
    """
    对单条数据进行查询优化的 RAG 流程。
    :param row: 单条数据的行，包含 query 和 answer
    :param max_iterations: 最大迭代次数
    :return: naive_answer, naive_rag_answer
    """
    query = row["query"]
    answer = row["answer"]

    key_words = extract_title(query)
    if not key_words:
        return None

    url, title, content = get_wikipedia_page(key_words)
    query_engine = embed_text(content[:10000])

    current_query = query
    for iteration in range(max_iterations):
        query_result = query_engine.query(current_query)
        retrieved_content = query_result.response

        feedback = feedback_evaluate(current_query, retrieved_content)

        if feedback == "fully relevant":
            # improved_rag_answer = llama_answer(question=query, external_knowledge=retrieved_content)
            improved_rag_answer = gpt_answer(
                question=query, external_knowledge=retrieved_content
            )

            return improved_rag_answer

        elif feedback == "partially relevant":
            current_query = adjust_query(current_query, feedback)

        elif feedback == "irrelevant":
            key_words = extract_title(current_query)
            if not key_words:
                return None

            url, title, content = get_wikipedia_page(key_words)
            query_engine = embed_text(content[:20000])

    # improved_rag_answer = llama_answer(question=query, external_knowledge=retrieved_content)
    improved_rag_answer = gpt_answer(
        question=query, external_knowledge=retrieved_content
    )

    return improved_rag_answer
