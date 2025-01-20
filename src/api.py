from fastapi import FastAPI, Query
from datetime import date
from contextlib import asynccontextmanager
import logging
from transformers import pipeline

from src.dto import TweetInput


# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting the app...")
    yield
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

# Endpoints

@app.post("/process_date/")
async def process_date(input_date: date = Query(..., example="2025-01-01")):
    """
    Endpoint to process a date input.
    """
    # Logic to handle the date input
    # TODO: Actually utilize our dataset to find for a given date.
    return {"message": f"Received date: {input_date}"}


@app.post("/process_tweet/")
async def process_tweet(tweet_input: TweetInput):
    """
    Endpoint to process a tweet-like text input.
    """
    # Initialize sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased")
    # Analyze sentiment
    sentiment = sentiment_analyzer(tweet_input.tweet)
    return {
        "message": f"Received tweet: {tweet_input.tweet}",
        "sentiment": sentiment
    }


# MAYDO: Add a /today endpoint which scrapes or fetches from API Elon's latest tweets.
