# RAGLLM

This project implement RAG technique to improve the performance of LLM. Retrieval-Augmented Generation (RAG) can help LLM avoid hallucication using provided documents. 

## Table of Contents

- [Features](#features)
- [Installation](#installation)


## Features

- Knowledge Base Retrieval: Retrieve relevant documents using all-mpnet-base-v2 embedded model and cosine similarity.

- Response Generation: Generate contextual answers using NVIDIA Gemini.

- Modular Design: Separate scripts for retrieval, generation, and workflow integration.

## Installation

1. Clone the Repository:

git clone https://github.com/luuducquy0510/RAGLLM.git
cd RAGLLM

2. Install Dependencies:

pip install -r requirements.txt

3. Set Up API Key:
Add your NVIDIA Gemini API key in scripts/generate.py:
