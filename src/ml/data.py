import os
import pandas as pd
from transformers import pipeline

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file containing the raw data.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    print(f"Looking for file at: {os.path.abspath(file_path)}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = pd.read_csv(file_path)
    print("Raw data loaded successfully!")
    return data

# Initialize the Hugging Face sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def compute_sentiment(text):
    """
    Compute sentiment for a given text.

    Parameters:
        text (str): The text to analyze.

    Returns:
        str: Sentiment label (e.g., "POSITIVE" or "NEGATIVE").
    """
    try:
        result = sentiment_analyzer(text[:512])  # Truncate to 512 tokens
        return result[0]['label']  # "POSITIVE" or "NEGATIVE"
    except Exception as e:
        print(f"Error processing text: {text}. Error: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess the raw data.

    Parameters:
        data (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    print("Preprocessing data...")
    # Keep only necessary columns
    data = data[['text', 'date']].copy()

    # Clean the text data
    data['text'] = data['text'].str.replace(r'http\S+', '', regex=True)  # Remove URLs
    data['text'] = data['text'].str.replace(r'@\w+', '', regex=True)     # Remove mentions
    data['text'] = data['text'].str.replace(r'\s+', ' ', regex=True).str.strip()  # Remove extra spaces

    # Parse the date and extract date features
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day

    # Drop the original 'date' column
    data.drop(columns=['date'], inplace=True)

    # Compute sentiment for each text
    print("Computing sentiment labels...")
    data['sentiment'] = data['text'].apply(compute_sentiment)

    print("Data preprocessing complete.")
    return data

def save_data(data, output_path):
    """
    Save the preprocessed data to a CSV file.

    Parameters:
        data (pd.DataFrame): Preprocessed data.
        output_path (str): Path to save the processed data.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directories if they don't exist
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {os.path.abspath(output_path)}")

# Main script execution
if __name__ == "__main__":
    raw_data_path = "data/raw/elon_musk_tweets.csv"
    processed_data_path = "data/processed/preprocessed_data.csv"  # Relative path for saving processed data

    print(f"Raw data path: {os.path.abspath(raw_data_path)}")
    print(f"Processed data path: {os.path.abspath(processed_data_path)}")

    # Load the raw data
    raw_data = load_data(raw_data_path)

    # Preprocess the data
    processed_data = preprocess_data(raw_data)

    # Save the preprocessed data
    save_data(processed_data, processed_data_path)
