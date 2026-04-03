# LLM Tokenizer & Dataset Pipeline (From Scratch + GPT-2)

##  Overview

This project implements a **basic tokenizer from scratch**, integrates **OpenAI GPT-2 tokenization using tiktoken**, and builds a **PyTorch dataset pipeline** for training language models.

It demonstrates the **complete preprocessing workflow** used in Large Language Models (LLMs):

* Text → Tokens → IDs
* Sliding window dataset creation
* Input-target sequence generation
* Token + positional embeddings

---

##  Features

###  1. Custom Tokenizer (SimpleTokenizerV2)

* Converts text into tokens using regex
* Handles unknown tokens using `<|unk|>`
* Supports decoding back to text
* Includes special token:

  * `<|endoftext|>`

---

###  2. Vocabulary Creation

* Unique tokens extracted from dataset
* Special tokens added:

  * `<|endoftext|>`
  * `<|unk|>`

 Vocabulary size increases after adding special tokens 

---

###  3. GPT-2 Tokenization (tiktoken)

* Uses GPT-2 tokenizer for real-world LLM compatibility
* Handles unseen words efficiently
* Supports special tokens

---

###  4. Dataset Creation for LLM Training

Implemented using PyTorch:

#### `GPTDatasetV1`

* Converts full text into token IDs
* Uses **sliding window approach**
* Generates:

  * Input sequence
  * Target sequence (shifted by 1)

---

###  5. DataLoader Pipeline

* Batch processing
* Shuffle & stride control
* Efficient training data generation

---

###  6. Embeddings

* Token embeddings
* Positional embeddings
* Combined input embeddings for transformer models

---

##  Project Structure

```
project/
│── tokenizer.ipynb
│── data/
│     └── the-verdict.txt
│── README.md
```

---

##  How It Works

### Step 1: Tokenization

```python
ids = tokenizer.encode("Hello world")
```

---

### Step 2: Sliding Window Dataset

```python
dataset = GPTDatasetV1(text, tokenizer, max_length=256, stride=128)
```

---

### Step 3: DataLoader

```python
dataloader = create_dataloader_v1(text)
```

---

### Step 4: Embeddings

```python
embedding = token_embedding_layer(input_ids)
```

---

##  Key Concepts Covered

* Tokenization (basic + GPT-2)
* Vocabulary mapping
* Special tokens in LLMs:

  * BOS (Beginning of Sequence)
  * EOS (End of Sequence)
  * PAD (Padding)
* Sliding window training
* Sequence-to-sequence prediction
* Embedding layers in transformers

---

##  Requirements

```bash
pip install torch tiktoken
```

---

##  Learning Outcomes

By completing this project, you will understand:

* How LLMs process raw text
* How tokenization affects model performance
* How training data is structured for transformers
* How embeddings are constructed

---

##  Future Improvements

* Add BPE tokenizer implementation
* Train a mini GPT model
* Add attention mechanism
* Build full transformer from scratch


