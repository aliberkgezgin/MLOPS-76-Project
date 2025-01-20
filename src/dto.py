from pydantic import BaseModel, Field
from datetime import date

# Define a model for the tweet input
class TweetInput(BaseModel):
    tweet: str = Field(..., example="This is a sample tweet!")


# Define a model for date input to avoid field clashes
class DateInput(BaseModel):
    input_date: date = Field(..., example="2025-01-01")
