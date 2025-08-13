import argparse
import pandas as pd
import os
from dataset import load_dataset
from refineloop import iterative_query_improvement
from pure_llm import gpt_answer
from naiverag import extract_title, embed_text
from get_wikipedia import get_wikipedia_page


def download_and_load_dataset(dataset_name="sentence-transformers/natural-questions"):
    """
    Download and load the dataset if it doesn't exist locally.
    :param dataset_name: The Hugging Face dataset name to load.
    :return: A pandas DataFrame of the dataset.
    """
    print(f"Downloading and loading dataset: {dataset_name}...")
    try:
        nq_ds = load_dataset(dataset_name)
        nq_train_df = pd.DataFrame(nq_ds["train"])
        print(f"Dataset '{dataset_name}' successfully loaded.")
        return nq_train_df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run the RAG pipeline with specified parameters."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["naive", "naive_rag", "refineloop"],
        default="refineloop",
        help="Choose the mode: naive, naive_rag, or refineloop.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations for query refinement.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to process."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.csv",
        help="File to save the results.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sentence-transformers/natural-questions",
        help="The dataset to use. Default: sentence-transformers/natural-questions",
    )
    args = parser.parse_args()

    nq_train_df = download_and_load_dataset(args.dataset)
    df = nq_train_df.head(args.num_samples)

    results = []
    for _, row in df.iterrows():
        query = row["query"]
        answer = row["answer"]
        if args.mode == "naive":
            naive_answer = gpt_answer(query)
            results.append(
                {"query": query, "answer": answer, "naive_answer": naive_answer}
            )
        elif args.mode == "naive_rag":
            key_words = extract_title(query)
            if key_words:
                _, _, content = get_wikipedia_page(key_words)
                query_engine = embed_text(content[:10000])
                query_result = query_engine.query(query)
                naive_rag_answer = gpt_answer(query, query_result.response)
                results.append(
                    {
                        "query": query,
                        "answer": answer,
                        "naive_rag_answer": naive_rag_answer,
                    }
                )
        elif args.mode == "refineloop":
            improved_answer = iterative_query_improvement(
                row, max_iterations=args.iterations
            )
            results.append(
                {"query": query, "answer": answer, "improved_answer": improved_answer}
            )

    print(f"Saving results to {args.output_file}...")
    pd.DataFrame(results).to_csv(args.output_file, index=False)
    print("Process completed!")


if __name__ == "__main__":
    main()
