import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    GPT_MODEL: str = "gpt-4"  # or "gpt-3.5-turbo"

settings = Settings()
