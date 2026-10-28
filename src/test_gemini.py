
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# Locate the project root and load the API key.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found. Check your .env file.")
else:
    client = None

    try:
        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": 30000},
        )

        print("Testing Gemini connection...", flush=True)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents="Reply with exactly: Gemini connection successful.",
        )

        print("Gemini response:")
        print(response.text or "Gemini returned an empty response.")

    except Exception as error:
        print("Gemini connection failed.")
        print(f"{type(error).__name__}: {error}")

    finally:
        if client is not None:
            client.close()
