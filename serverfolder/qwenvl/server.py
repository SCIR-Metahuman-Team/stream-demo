from qwenvl_chat_version2 import predict, asr_predict
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional
import uvicorn
import base64
import ipdb
app = FastAPI()


def encode_base64_str(imgfile):
    with open(imgfile, 'rb') as f:
        encoded_str = base64.b64encode(f.read())
    encoded_str = str(encoded_str)
    if encoded_str.startswith("b'") and encoded_str.endswith("'"):
        encoded_str = encoded_str[2:-1]
    return encoded_str
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


@app.post("/chat")
async def prediction(item: Item):
    return {"text": predict(message=item.message, img_base64=item.img_base64, history=item.history)}

@app.post("/asr")
def asr_prediction(asr_item: ASRItem):
    return {"asr_text": asr_predict(asr_item.asr_content)}

# test qwenvl
# def main():
#     message = 'Picture 1: <img>/home/tzheng2/workspace/scir-y1/demo_server/assets/f56d009bac9c060d8ba208697de0ed769f6a76fb/屏幕截图 2023-12-27 111223.png</img>\n请问这个是什么'
#     import ipdb
#     imgfile = '/home/tzheng2/workspace/scir-y1/demo_server/assets/20240726154218.png'
#     img_base64 = encode_base64_str(imgfile)
#     predict(message=message, img_base64=img_base64, history=[])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9988)
    # main()

