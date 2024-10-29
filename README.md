# Text2SQL - LLM from Scratch

This project implements a large language model (LLM) from scratch, designed to generate SQL queries from natural language inputs. 

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
Add links to the datasets and resources used for training. This section may include:
- [WikiSQL](https://paperswithcode.com/dataset/wikisql)
  

## Additional Resources
Provide any other useful tutorials, blog posts, or papers that can supplement learning during each project phase. These resources may include:
- Byte Pair Encoding (BPE) references.
- Attention mechanism tutorials.
- Decoding strategies and GPT model overviews.

