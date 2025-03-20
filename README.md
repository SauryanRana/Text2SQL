# Efficient Methods in Machine Learning: Text-to-SQL

## Project Overview

The goal of this project was to solve a language task using machine learning. The architecture was constrained by the condition that it has to be small enough to be executed on a consumer-grade laptop. The task we chose is the text-to-sql task, where natural language questions or instructions shall be translated into a valid sql query, which can then be used to query the database. Ideally this allows integration into a system that hides the underlying logic and simply allows to query a database using a natural language prompt.


## Contents


## Environment Setup

This section provides steps to set up an environment for this project.

### Prerequisites

Make sure to first clone this repository and navigate to it.

We recommend using conda for your environment management since it bundles a separate python installation within the environment for the project which removes the need to juggle different python versions on your system. You can install it via [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

If you decide against it make sure you have Python 3.12.4 installed and accessible via PATH.

1. **Create a new conda environment and install python:**
   ```bash
   conda create --name text2sql python=3.12.4
   ```
2. **Activate the environment:**
   ```bash
   conda activate text2sql
   ```
3. **Deactivate when done:**
   ```bash
   conda deactivate
   ```

### Setting Up the Environment

1. **If you want to be able to train models or need GPU-support for any other reason, make sure a Pytorch-compatible [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive) (11.8, 12.4 or 12.6) is installed:**
   ```bash
   nvcc --version
   ```
2. **Install the platform-specific pytorch version using pip according to [the official website](https://pytorch.org/get-started/locally/):**
   - CPU: 
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```
   - CUDA 11.8:
    ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   - CUDA 12.4:
    ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
   - CUDA 12.6:
    ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
3. **Install all remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start up a jupyter notebook client to play with our project:
   ```bash
   jupyter notebook
   ```

Use the inference.ipynb notebook to run inference on one of our pretrained models or on a model you trained yourself.

Use the train.ipynb notebook to create, train and save a model yourself.

Use the evaluation.ipynb notebook to run model inference on an entire dataset (validation data per default) and evaluate the models' performance.

## Dataset

We limited ourselves to the [WikiSQL](https://github.com/salesforce/WikiSQL) Dataset, which is limited to fairly simple queries that follow the structure ```SELECT [agg](col) FROM table [WHERE cond1 [AND cond2 [AND cond3]]]```.


## Model

We have built our Text-to-SQL model based on the [T5 architecture](https://arxiv.org/pdf/1910.10683), with a custom configuration that shrinks the model to under 1 Million Parameters.
