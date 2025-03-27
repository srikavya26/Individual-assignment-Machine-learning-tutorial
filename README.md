# Individual-assignment-Machine-learning-tutorial

## Sentiment Analysis with DistilBERT

### This repository contains a tutorial on performing sentiment analysis using DistilBERT and the Stanford Sentiment Treebank (SST-2) dataset. The tutorial provides a clear explanation of how to implement, train, and evaluate a sentiment analysis model using Hugging Face Transformers.
# Sentiment Analysis with DistilBERT

This repository contains a tutorial for performing sentiment analysis using DistilBERT and the Stanford Sentiment Treebank (SST-2) dataset. The tutorial covers essential concepts, model training, and evaluation using Hugging Face's Transformers library.


## Introduction
This project demonstrates how to use DistilBERT, a smaller and faster version of BERT, for sentiment analysis. The SST-2 dataset, containing movie review sentences labeled as positive or negative, is used for model training and evaluation.

## Requirements
- Python 3.8+
- Transformers (Hugging Face)
- PyTorch
- scikit-learn
- numpy

Install dependencies using:
```bash
pip install transformers torch scikit-learn numpy
```

## Getting Started
1. Clone the repository:
```bash
git clone https://github.com/srikavya26/Individual-assignment-Machine-learning-tutorial.git
cd sentiment-analysis-distilbert
```

2. Run the tutorial notebook:
```bash
jupyter notebook sst2_tutorial.ipynb
```

## Understanding DistilBERT
DistilBERT is a lightweight version of BERT, offering reduced size and faster inference without significantly compromising accuracy. It is ideal for deploying in resource-constrained environments.

## Attention Mechanism
The attention mechanism allows models like DistilBERT to focus on relevant words within a sentence. It computes the importance of words using query, key, and value matrices.


## Model Training and Evaluation
- **Dataset:** SST-2 (Stanford Sentiment Treebank)
- **Model:** DistilBERT (fine-tuned for sentiment analysis)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

Train and evaluate using the provided notebook.


## References
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Stanford Sentiment Treebank (SST-2)](https://nlp.stanford.edu/sentiment/index.html)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)



For contributions or issues, feel free to submit a pull request or open an issue.
