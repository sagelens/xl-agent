import sys
import os
from llama_cpp import Llama
CWD = os.getcwd()
def get_duckdb_sql(prompt):
    original_stderr = sys.stderr.fileno()
    saved_stderr = os.dup(original_stderr)

    # Redirect stderr to /dev/null
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, original_stderr)

    try:
        llama = Llama(
            model_path=CWD + "/models/duckdb-nsql.gguf",
            n_ctx=16384,
            n_threads=8,
            verbose=False
        )
    finally:
        # Restore stderr
        os.dup2(saved_stderr, original_stderr)
        os.close(devnull)
        os.close(saved_stderr)

    response = llama(prompt, temperature=0.1, max_tokens=700)
    generated_text = response["choices"][0]["text"]
    return generated_text
