from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Mail Assistant Running"}

@app.get("/hello/{name}")
def say_hello(name: str):
    return {"reply": f"Hello {name}"}