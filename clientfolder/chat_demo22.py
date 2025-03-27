# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

# 添加音频组件

from argparse import ArgumentParser
from pathlib import Path
import requests
import copy
import gradio as gr
import json
import os
from PIL import Image
from io import BytesIO
import re
from PIL import Image
import secrets
from datetime import datetime
import cv2
import json
import tempfile
import base64
import ipdb
from threading import Thread,Event
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
import time

with open('./config.json', 'r', encoding='utf-8') as f:
    args = json.load(f)
os.environ['GRADIO_TEMP_DIR'] = args['gradio']['temp_dir']
url = args['url']['chaturl']
url_asr = args['url']['asrurl']
url_image = args['url']['getvurl']
url_sgen = args['url']['isgenurl']
url_change_role = args['url']['chroleurl']
share = args['gradio']['share']
inbrowser = args['gradio']['inbrowser']
server_port = args['gradio']['port']
server_name = args['gradio']['host']
img_savepath = args['server']['img_savepath']
css = """
.container {
    max-width: 600px; /* 设置Gradio界面容器的最大宽度 */
    margin: 0 auto;  /* 使容器居中 */
}
"""
def encode_base64_str(imgfile):
    with open(imgfile, 'rb') as f:
        encoded_str = base64.b64encode(f.read())
    encoded_str = str(encoded_str)
    if encoded_str.startswith("b'") and encoded_str.endswith("'"):
        encoded_str = encoded_str[2:-1]
    return encoded_str

def encode_base64_str_direct(source_data):
    encoded_str = base64.b64encode(source_data)
    encoded_str = str(encoded_str)
    if encoded_str.startswith("b'") and encoded_str.endswith("'"):
        encoded_str = encoded_str[2:-1]
    return encoded_str

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

def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)
 
def _launch_demo():
    cap = cv2.VideoCapture(0)
    import atexit
    def on_exit():
        cap.release()
    atexit.register(on_exit)
    def predict(task_history):
        query = task_history[-1][0]
        print("User: " + _parse_text(query))
        history_cp = copy.deepcopy(task_history)
        full_response = ""

        history_filter = []
        pic_idx = 1
        pre = ""
        for i, (q, a) in enumerate(history_cp):
            if isinstance(q, (tuple, list)):
                q = f'Picture {pic_idx}: <img>{q[0]}</img>'
                pre += q + '\n'
                pic_idx += 1
            else:
                pre += q
                history_filter.append((pre, a))
                pre = ""
        history, message = history_filter[:-1], history_filter[-1][0]
        pattern = r'<img>(.*?)</img>'
        match = re.search(pattern, message)
        if match:
            file_path = message[match.span()[0] + 5: match.span()[1] - 6]
            start_index = re.search(r'assets/(.*?)', file_path).span()[1]
            
            file_path = file_path[start_index:]
            pre = args['gradio']['temp_dir']
            file_path = os.path.join(pre, file_path)
            encoding_string = encode_base64_str(imgfile=file_path)
        else:
            encoding_string = ''
        data = {
            'message': message,
            'history': history,
            'img_base64': encoding_string
        }
        response = requests.post(url, json=data)
        full_response = json.loads(response.text)['text']
        task_history[-1] = (query, full_response)
        print(full_response)
        return []
        
    def predict_stream(_chatbot, task_history):
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        print("User: " + _parse_text(query))
        history_cp = copy.deepcopy(task_history)
        full_response = ""

        history_filter = []
        pic_idx = 1
        pre = ""
        for i, (q, a) in enumerate(history_cp):
            if isinstance(q, (tuple, list)):
                q = f'Picture {pic_idx}: <img>{q[0]}</img>'
                pre += q + '\n'
                pic_idx += 1
            else:
                pre += q
                history_filter.append((pre, a))
                pre = ""
        history, message = history_filter[:-1], history_filter[-1][0]
        pattern = r'<img>(.*?)</img>'
        match = re.search(pattern, message)
        if match:
            file_path = message[match.span()[0] + 5: match.span()[1] - 6]
            pre = args['gradio']['temp_dir']
            file_path = '\\'.join(file_path.split('/')[-2:])
            file_path = os.path.join(pre, file_path)
            encoding_string = encode_base64_str(imgfile=file_path)
        else:
            encoding_string = ''
        data = {
            'message': message,
            'history': history,
            'img_base64': encoding_string
        }
        responses = requests.post(url, json=data, stream=True)
        for response in responses.iter_content(chunk_size=1024):
            response = response.decode('utf-8')
            print(response)
            _chatbot[-1] = (_parse_text(chat_query), _remove_image_special(_parse_text(response)))
            yield _chatbot
            full_response = _parse_text(response)
        response = full_response
        history.append((message, response))
        _chatbot[-1] = (_parse_text(chat_query), response)
        task_history[-1] = (query, full_response)
        print("Qwen-VL-Chat: " + _parse_text(full_response))
        yield _chatbot

    def change_role(choice):
        data = {
            'role': choice
        }
        requests.post(url_change_role, json=data)
        return []

    def capture_and_process_image():

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None

        ret, frame = cap.read()
        

        if not ret:
            print("Error: Failed to capture image.")
            return None

        return frame
    
    def add_text_asr(history, task_history, audio):
        
        now = datetime.now()
        file_name = now.strftime(os.path.join(args['gradio']['temp_dir'], "%Y%m%d%H%M%S.png"))
        cv2.imwrite(file_name, capture_and_process_image())
        history = history + [((file_name,), None)]
        new_filename = file_name
        global img_savepath
        print(new_filename)
        pattern = r'\\([^\\]+)$'
        match2 = re.search(pattern, new_filename)
        img_lastpath = new_filename[match2.span()[0]: match2.span()[1]][1:]
        img_saves = os.path.join(img_savepath, img_lastpath).replace('\\', '/')
        task_history = task_history + [((img_saves,), None)]
        audio_b64_str = encode_base64_str_direct(audio[1])
        data = {
            "asr_content": audio_b64_str
        }
        response = requests.post(url_asr, json=data)
        text = json.loads(response.text)['asr_text'][0]['text']
        text = re.sub(r'<.*?>', '', text)
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, gr.update(value=None)
    
    def add_text(history, task_history, message):
        for file_name in message["files"]:
            history = history + [((file_name,), None)]
            new_filename = file_name
            global img_savepath
            print(new_filename)
            pattern = r'\\([^\\]+\\[^\\]+)$'
            match2 = re.search(pattern, new_filename)
            img_lastpath = new_filename[match2.span()[0]: match2.span()[1]][1:]
            img_saves = os.path.join(img_savepath, img_lastpath).replace('\\', '/')
            task_history = task_history + [((img_saves,), None)]
        if message["text"] is not None:
            text = message["text"]     
            task_text = text
            if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
                task_text = text[:-1]
            history = history + [(_parse_text(text), None)]
            task_history = task_history + [(task_text, None)]
        return history, task_history, gr.MultimodalTextbox(value=None, interactive=False)

    def get_video(role, _chatbot, task_history):
        video_path = "./downloaded_video.mp4"
        text = task_history[-1][-1]
        data = {
            'role': role,
            'text': text
        }
        # def rqt_video(url_image, data):
            
        response = requests.post(url_image, json=data)
        
        if response.status_code == 200:
            # 获取视频内容
            video_content = response.content
            
            # 保存到文件
            with open(video_path, "wb") as f:
                f.write(video_content)
            print("视频下载完成")
        
        # requestthrd = Thread(target=rqt_video, args=(url_image, data))
        # requestthrd.start()
        # requests.post(url_sgen)
        ipdb.set_trace()
        # time.sleep(2)
        # for index in range(len(text)):
        #     time.sleep(0.2)
        #     _chatbot[-1][-1] = text[:index]
        #     yield _chatbot
        # _chatbot[-1][-1] = text
        return _chatbot, video_path
        

    with gr.Blocks(css=css) as demo:
        gr.Markdown("""
                    
# SCIR-SC 多模态X具身智能 虚拟数字人
""")
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label='video', height = 600)
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label='Chat History', bubble_full_width=False, elem_classes="control-height", height=450)
                chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="输入文字", show_label=False)
                audio_input = gr.Audio(sources="microphone", label="Record Audio")
                role_dropdown = gr.Dropdown(choices=["孙悟空", "nvbobaoyuan"], value="nvbobaoyuan")
                task_history = gr.State([])
                audio_input.stop_recording(add_text_asr, [chatbot, task_history, audio_input], [chatbot, task_history, audio_input]).then(
                    predict, [task_history], [], show_progress=True).then(
                        fn=get_video, inputs=[role_dropdown, chatbot, task_history], outputs=[chatbot, video]
                    )
                chat_input.submit(add_text, [chatbot, task_history, chat_input], [chatbot, task_history, chat_input]).then(
                    predict, [task_history], [], show_progress=True).then(
                    lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]).then(
                        fn=get_video, inputs=[role_dropdown, chatbot, task_history], outputs=[chatbot, video]
                    )
                role_dropdown.change(fn=change_role, inputs=[role_dropdown], outputs=[])
                gr.Examples(examples=[
                            {"files": [os.path.join(args['gradio']['temp_dir'], "examples/happy_example.png")], "text": "你觉得我现在心情怎么样？"},
                            {"files": [os.path.join(args['gradio']['temp_dir'], "examples/sad_example.png")], "text": "你觉得我现在心情怎么样？"},
                            ], inputs=chat_input, label="Image examples")
            
    demo.queue().launch(
        share=share,
        inbrowser=inbrowser,
        server_port=server_port,
        server_name=server_name
    )

def main():

    _launch_demo()

if __name__ == '__main__':
    main()
