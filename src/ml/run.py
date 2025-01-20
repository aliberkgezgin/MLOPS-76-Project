import os
from MLOPS_76_PROJECT.data import load_data, preprocess_data, save_data
from MLOPS_76_PROJECT.train import split_data, train_model

def main():
    """
    Main function to preprocess data, split it, and train the model.
    """
    # Paths
    raw_data_path = "data/raw/elon_musk_tweets.csv"
    processed_data_path = "data/processed/preprocessed_data.csv"
    model_save_path = "models/sentiment-analysis-model"
    model_name = "distilbert-base-uncased"

    # Step 1: Preprocess Data
    if not os.path.exists(processed_data_path):
        print("Loading and preprocessing data...")
        raw_data = load_data(raw_data_path)
        processed_data = preprocess_data(raw_data)
        save_data(processed_data, processed_data_path)
        print(f"Data preprocessed and saved to: {processed_data_path}")
    else:
        print(f"Using existing preprocessed data: {processed_data_path}")

    # Step 2: Split Data
    print("Splitting data into training and validation sets...")
    train_dataset, val_dataset = split_data(processed_data_path)
    print("Data splitting complete.")

    # Step 3: Train Model
    print("Starting model training...")
    train_model(model_name, train_dataset, val_dataset, model_save_path)
    print(f"Model training complete. Model saved to: {model_save_path}")

if __name__ == "__main__":
    main()

