# Text-to-SQL Query Generator

## Project Overview

This goal of this project it to develop a LLM from scratch that can translate text-based questions into SQL queries. Such a model is useful in scenarios where users need to interact with databases using natural language, making data access more intuitive. The project will follow a structured timeline, focusing on data collection, preprocessing, model setup, and iterative fine-tuning.


## Contents

1. **Working with Text Data**  
   - Develop tokenization and encoding functions, including special tokens and BPE.
2. **Implementing Attention Mechanisms**  
   - start building of attention mechanisms, critical for efficient LLM function.
3. **Building a GPT Model from Scratch**  
   - Coding and training a GPT model specifically for text generation.
4. **Training the LLM on Labeled and Unlabeled Data**  
   - Techniques for model evaluation and understanding decoding strategies like temperature and top-k sampling.
5. **Fine-Tuning to Follow Instructions (Optional)**  
   - Fine-tuning to enhance model accuracy and usability, especially for specific instructions.

## Environment Setup

This section provides steps to set up a conda environment for this project.

### Prerequisites
- Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Ensure Python version 3.12.4 is available.

### Setting Up the Environment
1. **Create a new conda environment:**
   ```bash
   conda create -n text2sql python=3.12.4
   ```
2. **Activate the environment:**
   ```bash
   conda activate text2sql
   ```
3. **Deactivate when done:**
   ```bash
   conda deactivate
   ```

## Data Sources

The datasets we are using include:

- **WikiSQL Dataset**: A supervised dataset for natural language to SQL mapping.
- **Spider Dataset**: A large-scale Text-to-SQL dataset.
- **Custom Datasets**: Any additional data gathered to meet specific project needs.
- **Awesome Text2SQL** - https://github.com/eosphoros-ai/Awesome-Text2SQL

## Model

We are building our Text-to-SQL model based on the GPT architecture. This architecture is suited for generating structured outputs like SQL queries from natural language inputs.

### Model Specifications

- **Architecture**: GPT-based, focusing on a decoder-only structure for text-to-SQL generation.
- **Frameworks**: PyTorch or TensorFlow, depending on team preferences.
- **Embedding**: SQL-specific embeddings and carefully crafted prompts to improve SQL generation accuracy.

## Learning Resources

### Primary Reference

- **Book**: *Build a Large Language Model from Scratch* by Sebastian Raschka
  - This book will guide our process from data collection and preprocessing to model architecture and evaluation.

### Additional Resources

- **Hugging Face Transformers Documentation**: For understanding and implementing GPT architecture.
- **SQL Syntax Documentation**: For structuring and validating SQL queries generated by the model.


