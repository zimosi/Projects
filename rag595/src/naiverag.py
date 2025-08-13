from openai import OpenAI
from typing import List
from llama_index.core import VectorStoreIndex, Document

from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None


def extract_title(question: str) -> str:
    """
    Extract key words from a given question using OpenAI GPT or similar NLP techniques.

    :param question: The input question as a string.
    :return: A list of extracted keywords.
    """
    client = OpenAI()
    prompt = f"""
        You are a knowledgeable and concise assistant. I will provide you with a question, 
        and your task is to identify the most relevant Wikipedia article title where I can 
        find a correct and comprehensive answer to this question.

        Please provide your response as a single title.

        Here is the question:
        "{question}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        output = response.choices[0].message.content.strip()
        # Split the response into a list of keywords
        return output
    except Exception as e:
        print(f"Error: {e}")
        return ""


def chunk_text(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def embed_text(long_text, chunk_size=500, chunk_overlap=20, top_k=10):
    text_chunks = chunk_text(long_text, chunk_size, chunk_overlap)

    chunked_documents = [Document(text=chunk) for chunk in text_chunks]

    index = VectorStoreIndex.from_documents(chunked_documents)
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    return query_engine


# long_text = """The University of California, Berkeley (UC Berkeley, Berkeley, Cal, or California)[10][11] is a public land-grant research university in Berkeley, California, United States. Founded in 1868 and named after the Anglo-Irish philosopher George Berkeley, it is the state's first land-grant university and is the founding campus of the University of California system.[12]

# Berkeley has an enrollment of more than 45,000 students. The university is organized around fifteen schools of study on the same campus, including the College of Chemistry, the College of Engineering, and the Haas School of Business. It is classified among "R1: Doctoral Universities – Very high research activity".[13] The Lawrence Berkeley National Laboratory was originally founded as part of the university.[14]

# Berkeley was a founding member of the Association of American Universities and was one of the original eight "Public Ivy" schools. In 2021, the federal funding for campus research and development exceeded $1 billion.[15] Thirty-two libraries also compose the Berkeley library system which is the sixth largest research library by number of volumes held in the United States.[16][17][18]

# """

# # 嵌入并创建查询引擎
# query_engine = embed_text(long_text)

# # 示例查询
# query_result = query_engine.query("berkeley")
# print("查询结果:", query_result)


# response will be a combined response from all the chunks. 经过二次创作
# response = query_engine.query("U.S. News")
# print("Final Answer:", response.response)

# # 查看所有被检索到的文本块
# for i, node_with_score in enumerate(response.source_nodes):
#     print("------")
#     print(f"Chunk #{i+1} (Similarity Score: {node_with_score.score}):")
#     print(node_with_score.node.text)
#     print("------")
