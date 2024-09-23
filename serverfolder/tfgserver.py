import argparse
import ipdb
import json
import os
import sys
import importlib
import torch
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from contextlib import asynccontextmanager

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
    text: str
    role: str

class CRole(BaseModel):
    role: str

current_dir = os.path.dirname(os.path.abspath(__file__))

def getConfig(configPath) -> dict:
    with open(configPath, 'r', encoding='utf-8') as f:
        configDict = json.load(f)
    return configDict

class TfgServer():
    def __init__(self, config):
        global current_dir
        sys.path.append(os.path.join(current_dir, 'models'))
        modulename = f'{config["tfgmodel"]}.main'
        tfgmodule = importlib.import_module(modulename)
        self.model = getattr(tfgmodule, "mainmodel")

configDict = getConfig(os.path.join(os.path.dirname(current_dir), 'config.json'))
tfgConfig = configDict['tfg']
# parser = argparse.ArgumentParser()
# parser.add_argument("--tfgmodel", required=False, type=str, default=tfgConfig['tfgmodel'])
# args = parser.parse_args()

tfgsv = TfgServer(tfgConfig)

@app.post("/changerole")
def changerole(crole: CRole):
    tfgsv.model.changerole(crole.role)

@app.post("/sgen")
def is_startgen():
    tfgsv.model.is_startgen()

@app.post("/getv")
def videogen(item: Item):
    tfgsv.model.videogen(item.role, item.text)