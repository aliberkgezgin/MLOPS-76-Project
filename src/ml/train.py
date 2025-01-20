import logging
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation during training.

    Parameters:
        eval_pred (tuple): Tuple containing predictions and labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def split_data(file_path, test_size=0.2):
    """
    Load the preprocessed data and split it into training and validation datasets.

    Parameters:
        file_path (str): Path to the preprocessed CSV file.
        test_size (float): Proportion of the dataset to use for validation.

    Returns:
        tuple: Training and validation datasets.
    """
    logging.info(f"Loading preprocessed data from {file_path}...")
    data = pd.read_csv(file_path)

    # Map sentiment values to numerical labels
    logging.info("Mapping 'sentiment' column to numerical labels...")
    if "sentiment" in data.columns:
        data["label"] = data["sentiment"].map({"POSITIVE": 1, "NEGATIVE": 0})
    else:
        raise ValueError("The dataset must include a 'sentiment' column.")

    dataset = Dataset.from_pandas(data)

    # Split the dataset
    logging.info("Splitting data into training and validation sets...")
    split_dataset = dataset.train_test_split(test_size=test_size)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    return train_dataset, val_dataset

def train_model(model_name, train_dataset, val_dataset, save_path="./models/sentiment-analysis-model"):
    """
    Fine-tune a pre-trained Hugging Face model on the training dataset.

    Parameters:
        model_name (str): Name of the pre-trained model.
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        save_path (str): Path to save the fine-tuned model and tokenizer.
    """
    # Load tokenizer and model
    logging.info(f"Loading pre-trained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize datasets
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    logging.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Ensure labels are available
    if "label" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("label", "labels")
        val_dataset = val_dataset.rename_column("label", "labels")
    else:
        raise ValueError("The dataset does not contain a 'label' column for training.")

    # Remove unnecessary columns and format datasets for PyTorch
    columns_to_remove = ["text", "year", "month", "day", "sentiment"]
    logging.info(f"Removing columns: {columns_to_remove}")
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    val_dataset = val_dataset.remove_columns(columns_to_remove)

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # Define training arguments
    logging.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Attach custom metric function
    )

    # Train the model
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training complete.")

    # Save the fine-tuned model
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logging.info(f"Model and tokenizer saved to {save_path}")
