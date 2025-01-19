# data.py
# This script is responsible for loading, cleaning, and preprocessing the Elon Musk tweets dataset.
# The goal is to prepare the data for downstream analysis or modeling by retaining only the necessary columns,
# cleaning the text data, parsing date information into separate features, and saving the preprocessed data to a file.

import pandas as pd
import re
import os
from datetime import datetime

def load_data(file_path):
    """
    Load the dataset from a specified file path into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CSV file containing raw data.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the data by cleaning text, extracting relevant columns, and parsing date information.

    Parameters:
        data (pd.DataFrame): The raw data DataFrame.

    Returns:
        pd.DataFrame: Preprocessed data as a pandas DataFrame.
    """
    # Keep only the 'text' and 'date' columns
    data = data[['text', 'date']]

    # Define a function to clean text data
    def clean_text(text):
        """
        Remove URLs, mentions (starting with @), and extra whitespace from the text.

        Parameters:
            text (str): Input tweet text.

        Returns:
            str: Cleaned text.
        """
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)    # Remove mentions starting with @
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text

    # Apply the cleaning function to the 'text' column
    data['text'] = data['text'].apply(clean_text)

    # Parse the 'date' column into new features
    def parse_date(date_str):
        """
        Parse the date string into year, month, day_of_month, day_of_week, and hour_of_day.

        Parameters:
            date_str (str): Input date string.

        Returns:
            dict: Dictionary containing parsed date components.
        """
        date_obj = datetime.fromisoformat(date_str.replace('Z', ''))
        return {
            'year': date_obj.year,
            'month': date_obj.month,
            'day_of_month': date_obj.day,
            'day_of_week': date_obj.isoweekday(),  # Monday=1, Sunday=7
            'hour_of_day': date_obj.hour
        }

    # Apply the date parsing function and create new columns
    date_parsed = data['date'].apply(parse_date)
    date_df = pd.DataFrame(date_parsed.tolist())
    data = pd.concat([data, date_df], axis=1)

    # Drop the original 'date' column
    data.drop(columns=['date'], inplace=True)

    return data

def save_preprocessed_data(data, output_path):
    """
    Save the preprocessed data to a specified file path as a CSV file.

    Parameters:
        data (pd.DataFrame): The preprocessed data DataFrame.
        output_path (str): Path to save the preprocessed CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
    data.to_csv(output_path, index=False)

# Main execution
if __name__ == "__main__":
    # Define file paths
    raw_data_path = r"C:\Users\45502\Desktop\MLOPS\MLOPS-76-Project\data\raw\elon_musk_tweets.csv"
    preprocessed_data_path = r"C:\Users\45502\Desktop\MLOPS\MLOPS-76-Project\data\preprocessed\clean_data.csv"

    # Load the raw data
    print("Loading data...")
    raw_data = load_data(raw_data_path)

    # Preprocess the data
    print("Preprocessing data...")
    preprocessed_data = preprocess_data(raw_data)

    # Save the preprocessed data
    print("Saving preprocessed data...")
    save_preprocessed_data(preprocessed_data, preprocessed_data_path)

    # Print a small sample of the preprocessed data
    print("Sample of preprocessed data:")
    print(preprocessed_data.head())
