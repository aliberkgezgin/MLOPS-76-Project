# MLOPS-76-Project
This is the repository for the final project of 02476 MLOPS course of DTU for Winter 2025 period.

Group 76 members:
- Lars Kæraa Lücke
- Ali Berk Gezgin
- Jónas Ingi Valdimarsson
- Charalampos Stathopoulos
- Matteo D'Souza

# **Description of the Project**

Welcome to our MLOps project for classifying Elon Musk’s mood based on his Tweets. In this project, we will perform sentiment analysis on text data from Twitter. Our main goal is to learn the basics of MLOps while building a simple model that can rate and categorize Elon Musk’s emotional tone.

## **1. Dataset:**

We plan to work with a collection of Elon Musk’s tweets. We will gather these from Twitter, focusing on recent posts or a chosen timeframe. This dataset will include each tweet’s text and the date and other useful metadata. Our aim is to keep it straightforward, so we will mainly use the text as input.

## **2. Model:**

For this project, we are planning to work with a pre-trained model called DistilBERT. It is a smaller and faster version of BERT, which is a popular language model for many NLP tasks. DistilBERT is good for sentiment analysis because it is both accurate and quick to run.

## **3. PyTorch-Based Third-Party Package:**

We will rely on the Hugging Face Transformers library. This library supports PyTorch and gives us many pre-trained models, including DistilBERT. It also provides helpful tools for tokenization and fine-tuning.

## **4. Project Goal:**

The main goal is to predict the sentiment or mood (positive, negative, or neutral) of Elon Musk’s tweets. By doing this, we hope to learn how to train, deploy, and manage machine learning models in an MLOps setting. We also want to practice best practices for code organization and version control.

## **5. MLOps Frameworks:**

We intend to use a few simple frameworks. First, we will use MLflow for experiment tracking, so we can keep records of our runs, hyperparameters, and metrics. Second, we will use FastAPI to create a simple REST API for model predictions. We may also use Docker to containerize our application for easy deployment. Everything will be tracked with Git for version control.

## **6. Data:**

Our data will be the text from Elon Musk’s tweets. We will run them through the DistilBERT model for sentiment analysis. This text data is easy to collect from Twitter with public APIs or already available datasets from Kaggle.

## **7. Expected Models:**

We plan to use DistilBERT for sentiment classification. Later, if we want more advanced analysis, we could explore other Transformer models, but we will start with DistilBERT due to its simplicity and speed.
