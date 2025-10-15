# Excel Natural Language Query Agent

This project provides an on-device agent that ingests messy, real-world Excel workbooks, materializes them into a clean database representation, and allows users to ask questions in plain English.

## Configuration

The agent can be configured to use one of two runtimes for Natural Language to SQL translation. Edit the `config.yaml` file to choose your preferred runtime.

```yaml
# config.yaml

# Set to 'true' to use the Ollama runtime.
# Set to 'false' to use a local GGUF model with llama-cpp-python runtime.
USE_OLLAMA: true
```

## Setup Instructions

1. Prerequisites

    Python 3.10+

    (Optional) Ollama if you plan to use it as the backend (USE_OLLAMA: true).

2. Clone and Set Up Environment

Clone the repository, navigate into the directory, and set up a Python virtual environment.

```Bash

# For macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install Dependencies

Install all the required Python packages from the requirements.txt file.
Bash

`pip install -r requirements.txt`

Note for Mac Users: For better performance on Apple Silicon, you may need to install llama-cpp-python with Metal support.

`CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python`

4. Model Setup

You only need to follow the instructions for the backend you chose in config.yaml.

#### Option A: Using Ollama (USE_OLLAMA: true)

Install Ollama: Download and run the installer from the official Ollama website.

Pull the Model: Open your terminal and pull the required `duckdb-nsql` model. The application will connect to local Ollama server automatically.

`ollama pull duckdb-nsql`

### Option B: Using Local GGUF Model (USE_OLLAMA: false)

This will automatically download the GGUF from HF_HUB and start pipeline (note that this adds observable latency in SQL generation)

5. How to Run

Assuming you have an app.py with a Streamlit interface.

- Ensure virtual environment is activated.

- Start the Streamlit application from project's root directory.

`streamlit run app.py`

The application GUI should now be accessible in browser, typically at http://localhost:8501.

- Use this test workbook: `/data/search.xlsx`

## Test Questions


- Core SQL & Joins

    - How many shipments were made by FedEx?

    - List the top 3 departments by total spend.

- Safety & Validation

    - DELETE FROM logistics_0 WHERE carrier = 'FedEx'

    - Show me shipments from February 30, 2025

    - What is the average discount by category?

    - Show me the revenue.

    - What are the top 2 categories by gross revenue?

    - What was the total net revenue?

    - List all shipments with a package weight over 200 lbs.

    - What was the total net revenue for the 'Services' category?

    - What was the total gross revenue for the 'Hardware' category?

    - (Empty query to test input handling)

