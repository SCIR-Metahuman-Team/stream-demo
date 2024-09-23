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
current_dir = os.path.dirname(os.path.abspath(__file__))

def getConfig(configPath) -> dict:
    with open(configPath, 'r', encoding='utf-8') as f:
        configDict = json.load(f)
    return configDict

class TtsServer():
    def __init__(self, config):
        global current_dir
        sys.path.append(os.path.join(current_dir, 'models'))
        modulename = f'{config["ttsmodel"]}.main'
        ttsmodule = importlib.import_module(modulename)
        self.model = getattr(ttsmodule, "mainmodel")
        
configDict = getConfig(os.path.join(os.path.dirname(current_dir), 'config.json'))
tfgConfig = configDict['tts']
# parser = argparse.ArgumentParser()
# parser.add_argument("--ttsmodel", type=str, required=False, default=tfgConfig['ttsmodel'])
# args = parser.parse_args()
ttssv = TtsServer(tfgConfig)

@app.post("/fish")
def get_audio(item: Item):
    ttssv.model.get_audio(item.role, item.text)