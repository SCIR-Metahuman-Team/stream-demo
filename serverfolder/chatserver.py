from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
import sys
import importlib
import argparse
import torch
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional


app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    message: str
    history: Optional[List[Tuple[str, str]]] = None
    img_base64: str

class ASRItem(BaseModel):
    asr_content: str

current_dir = os.path.dirname(os.path.abspath(__file__))

def getConfig(configPath) -> dict:
    with open(configPath, 'r', encoding='utf-8') as f:
        configDict = json.load(f)
    return configDict

class ChatServer():
    def __init__(self, config):
        global current_dir
        self.config = config
        sys.path.append(os.path.join(current_dir, 'models'))
        modulename = f'{config["chatmodel"]}.main'
        chatmodule = importlib.import_module(modulename)
        self.model = getattr(chatmodule, "mainmodel")

configDict = getConfig(os.path.join(os.path.dirname(current_dir), 'config.json'))
chatConfig = configDict['chat']
# parser = argparse.ArgumentParser()
# parser.add_argument("--chatmodel", required=False, type=str, default=chatConfig['chatmodel'])
# args = parser.parse_args()

chatsv = ChatServer(chatConfig)

@app.post("/chat")
async def predict(item: Item):
    return chatsv.model.predict(item.message, item.img_base64, item.history)

@app.post("/asr")
def asr_predict(asr_item: ASRItem):
    return chatsv.model.asr_predict(asr_item.asr_content)
