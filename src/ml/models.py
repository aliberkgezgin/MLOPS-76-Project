from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

def create_model(model_name="distilbert-base-uncased", num_labels=2):
    """
    Create and load a pre-trained model and tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def save_model(model, tokenizer, save_path="./models/sentiment-analysis-model"):
    """
    Save the model and tokenizer.
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")

def load_saved_model(save_path="./models/sentiment-analysis-model"):
    """
    Load a saved model and tokenizer.
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model directory '{save_path}' does not exist.")
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    return model, tokenizer
