from fish_speech_demo import get_audio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
import sys
import importlib
import ipdb


app = FastAPI()

class Item(BaseModel):
    text: str
    role: str

@app.post("/fish")
def get_web_audio(item: Item):
    get_audio(item.role, item.text)

# def main():
#     get_audio('Klee', '大家好，我叫可莉')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
    # main()
