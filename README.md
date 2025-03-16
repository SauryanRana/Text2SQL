# Text-to-SQL Query Generator

## Project Overview

This goal of this project it to develop a LLM from scratch that can translate text-based questions into SQL queries. Such a model is useful in scenarios where users need to interact with databases using natural language, making data access more intuitive. The project will follow a structured timeline, focusing on data collection, preprocessing, model setup, and iterative fine-tuning.

## Environment Setup

This section provides steps to set up a conda environment for this project.

### Prerequisites
- Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Ensure Python version 3.12.4 is available.

### Setting Up the Environment
1. **Create a new conda environment:**
   ```bash
   conda create -n t2sql python==3.12.4
   ```
2. **Activate the environment:**
   ```bash
   conda activate t2sql
   ```
3. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the preprocessed data:**
   Download the required data files from [Google Drive](https://drive.google.com/drive/folders/1Ep17N1jNFVyJYfwkMCJqx8lpEjBg_4-d?usp=sharing) and place them in the following structure:
   ```
   Text2SQL/
   └── data/
       └── preprocessed_wikisql/       # Preprocessed data files
   ```
5. **Deactivate when done:**
   ```bash
   conda deactivate
   ```

### Running the Model

```bash
# For base custom T5 model
python -m src.main --model base

# For efficient-tiny T5 model
python -m src.main --model efficient
```

### Project Structure

```bash
├── Text2SQL/
│   ├── data/
│   │   └── preprocessed_wikisql/     # Preprocessed WikiSQL dataset
│   │
│   ├── src/
│   │   ├── __init__.py              # Package initialization
│   │   ├── main.py                  # Entry point for training and inference
│   │   ├── model.py                 # Model architecture definitions
│   │   ├── dataset.py               # Dataset loading and processing
│   │   ├── train.py                 # Training loop implementation
│   │   ├── eval.py                  # Evaluation functions
│   │   ├── tokenizer.py             # Tokenization utilities
│   │   └── utils.py                 # Helper functions and utilities
│   │
│   ├── .gitignore                   # Git ignore rules
│   ├── README.md                    # Project documentation
│   └── requirements.txt             # Python dependencies
```
