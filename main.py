from fastapi import FastAPI, HTTPException
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI()

# Configure Groq
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# AI Route
@app.get("/ask_ai")
def ask_ai(question: str):

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": question}
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "answer": response.choices[0].message.content
    }
