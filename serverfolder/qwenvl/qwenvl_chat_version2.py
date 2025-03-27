from argparse import ArgumentParser
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import List, Tuple, Optional
import torch
import uvicorn
import base64
from pydantic import BaseModel
import os
import ipdb
import wave
import json
import re
# uvicorn qwenvl_chat:app --host "0.0.0.0" --port 9880

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
asr_saves = os.path.join(root_dir, 'serverfolder/assets/temp.wav')
with open(os.path.join(root_dir, "config.json"), 'r', encoding='utf-8') as f:
    confarg = json.load(f)

cpu_only = False
checkpoint_path = "/home/tzheng2/workspace/scir-y1/QwenVL/output_merged03"
dev = False

def decode_base64_str(encoded_str, save_path):
    if encoded_str.startswith("b'") and encoded_str.endswith("'"):
        encoded_str = encoded_str[2:-1]
    decoding_bytes = base64.b64decode(encoded_str)
    folder_path = os.path.dirname(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(save_path, 'wb') as f:
        f.write(decoding_bytes)

def wav_decode_base64_str(encoded_str, save_path):
    if encoded_str.startswith("b'") and encoded_str.endswith("'"):
        encoded_str = encoded_str[2:-1]
    decoding_bytes = base64.b64decode(encoded_str)
    folder_path = os.path.dirname(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with wave.open(save_path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(decoding_bytes)

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

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)

DEFAULT_CKPT_PATH = 'Qwen/Qwen-VL-Chat'

asr_inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/SenseVoiceSmall',
    model_revision="master")


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    args = parser.parse_args()
    return args

def _load_model_tokenizer(cpu_only, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen-VL-Chat', trust_remote_code=True, resume_download=True, revision='master',
    )

    if cpu_only:
        device_map = "cpu"
    else:
        device_map = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
        revision='master',
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        checkpoint_path, trust_remote_code=True, resume_download=True, revision='master',
    )

    return model, tokenizer

if not dev:
    model, tokenizer = _load_model_tokenizer(cpu_only, checkpoint_path)
# file: UploadFile = File(...),
#  history: List[str] = Form(...)

class Item(BaseModel):
    message: str
    history: Optional[List[Tuple[str, str]]] = None
    img_base64: str

class ASRItem(BaseModel):
    asr_content: str

def generate_stream(tokenizer, message, history):
    for response in model.chat_stream(tokenizer, message, history=history):
        print(response)
        yield response

def predict(message, img_base64, history):
    # 'Picture 1: <img>/home/tzheng2/workspace/scir-y1/demo_server/assets/f56d009bac9c060d8ba208697de0ed769f6a76fb/屏幕截图 2023-12-27 111223.png</img>\n请问这个是什么'
    pattern = '<img>(.*?)</img>'
    match = re.search(pattern, message)
    if match:
        img_saves = message[match.span()[0] + 5: match.span()[1] - 6]
        decode_base64_str(img_base64, img_saves)
    result = ''
    for response in generate_stream(tokenizer, message, history):
        result = response
        continue
    return result

    # return StreamingResponse(generate_stream(tokenizer, message, history), media_type='text/event-stream')

def asr_predict(asr_content):
    wav_decode_base64_str(asr_content, asr_saves)
    rec_result = asr_inference_pipeline(asr_saves)
    return rec_result
