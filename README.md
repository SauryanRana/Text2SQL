# Text-to-SQL Query Generator

## Project Overview

This goal of this project it to develop a LLM from scratch that can translate text-based questions into SQL queries. Such a model is useful in scenarios where users need to interact with databases using natural language, making data access more intuitive. The project will follow a structured timeline, focusing on data collection, preprocessing, model setup, and iterative fine-tuning.


## Contents


## Environment Setup

This section provides steps to set up a conda environment for this project.

### Prerequisites
- Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Ensure Python version 3.12.4 is available.

### Setting Up the Environment
1. **Create a new conda environment:**
   ```bash
   conda create -n text2sql python==3.12.4
   ```
2. **Activate the environment:**
   ```bash
   conda activate text2sql
   ```
3. **Deactivate when done:**
   ```bash
   conda deactivate
   ```

4. **Download the preprocessed data (if not included)**
   link


5. **Start Training**
   ```
   # For base custom T5 model
   python -m src.main --model base

   # For efficient-tiny T5 model
   python -m src.main --model efficient
   ```



## Data Sources

The datasets we are using include:

- **WikiSQL Dataset**: [Data](https://github.com/salesforce/WikiSQL) ,[GD Link ](https://drive.google.com/drive/folders/1I9AiEDlMb_3KWWfLeVvfjHVK87h3nGHo?usp=drive_link), [Collab](https://colab.research.google.com/drive/1dOTP5Fir04MLDD0nS8YpenwnWp8uG-de?usp=sharing)
- **Spider Dataset**: A large-scale Text-to-SQL dataset.
- **Custom Datasets**: Any additional data gathered to meet specific project needs.


### Repo structure to follow

```bash

├── Text2SQL/
│   ├── data/
│   │   ├── WikiSQL/                     # Raw datasets go here
│   │   └── preprocessed_wikisql/               # Processed data (e.g., tokenized) 
│   │
│   ├── src/
│   │   ├── __init__.py              # Package initialization
│   │   ├── train.py                 # Main training script
│   │   ├── evaluate.py              # Evaluation script
│   │   ├── data_processing.py       # Data loading and preprocessing
│   │   ├── model.py                 # Model initialization and setup
│   │   ├── utils.py                 # Utility functions (e.g., logging)
│   │   └── custom_layers.py         # (Optional) Custom model layers if any
│   │
│   ├── logs/
│   │   └── training_logs/           # Training logs, output from Accelerate, etc.
│   │
│   ├── outputs/
│   │   ├── checkpoints/             # Model checkpoints
│   │   └── metrics/                 # Evaluation metrics (e.g., loss, accuracy)
│   │
│   │
│   ├── .gitignore                   # Ignore files like checkpoints, data
│   ├── README.md                    # Project documentation
│   └── requirements.txt             # Python dependencies
```
