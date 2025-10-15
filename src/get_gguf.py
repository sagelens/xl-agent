from huggingface_hub import hf_hub_download

def download_(repo_id="apurv0405/duckdb-nsql-gguf",
    filename="duckdb-nsql.gguf",
    local_dir="./models/"):

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir
    )

    print(f"Downloaded to: {model_path}")
