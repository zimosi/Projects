
## Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

## Set up your API keys:
- **Hugging Face API Key**: Replace `YOUR_API_KEY` in the code with your Hugging Face API key.
- **OpenAI API Key**: Set the OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY=your_openai_api_key
```

## Usage
### Running the Pipeline
Run the pipeline using the `main.py` script. You can specify the mode, number of iterations, number of samples, and output file.

#### Example Commands:
- Run the naive pipeline:
```bash
python main.py --mode naive --num_samples 100 --output_file naive_results.csv
```

- Run the naive RAG pipeline:
```bash
python main.py --mode naive_rag --num_samples 100 --output_file naive_rag_results.csv
```

- Run the iterative RAG pipeline:
```bash
python main.py --mode refineloop --iterations 3 --num_samples 100 --output_file refineloop_results.csv
```

#### Parameters:
- `--mode`: The mode of operation. Options: `naive`, `naive_rag`, `refineloop`. Default: `refineloop`.
- `--iterations`: Number of iterations for query refinement (only for `refineloop` mode). Default: `1`.
- `--num_samples`: Number of samples to process. Default: `100`.
- `--output_file`: File to save the results. Default: `results.csv`.
- `--dataset`: The dataset to use. Default: `sentence-transformers/natural-questions`.

## Dataset
The default dataset used is **Natural Questions**, hosted on Hugging Face. If the dataset is not available locally, it will be automatically downloaded when the script runs.

### Custom Dataset
You can specify a custom dataset by passing the `--dataset` argument with the name of the Hugging Face dataset:
```bash
python main.py --dataset your-dataset-name --mode naive --num_samples 500 --output_file custom_results.csv
```

## Project Structure
The project consists of the following files and modules:
- `main.py`: The main entry point for running the pipeline with various modes and parameters.
- `dataset.py`: Handles loading and downloading the dataset.
- `get_wikipedia.py`: Contains functions to retrieve and parse content from Wikipedia.
- `pure_llm.py`: Implements naive LLM and naive RAG pipelines.
- `naive_rag.py`: Implements the naive RAG pipeline with Wikipedia content retrieval.
- `refineloop.py`: Implements the iterative query refinement process for the RAG pipeline.
- `answer_eval.py`: Evaluates the equivalence of generated answers and calculates similarity scores.

## Results
The pipeline outputs the results as a CSV file, with columns for queries, answers, and generated responses. The results vary depending on the chosen mode (`naive`, `naive_rag`, or `refineloop`).
