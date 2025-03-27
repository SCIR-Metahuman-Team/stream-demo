from glob import glob
import shutil
import torch
from time import  strftime
import ffmpeg
import os, sys, time
import uvicorn
import json
import cv2
from queue import Queue
from threading import Thread,Event
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from contextlib import asynccontextmanager
import base64
from pydantic import BaseModel
import requests
import ipdb
import threading
from threading import Thread,Event
import asyncio
from sadtalker_web import l_batch_main
# from rtmp_streaming import StreamerConfig, Streamer
url = 'http://127.0.0.1:9999/fish'


app = FastAPI()
def decode_base64_str(encoded_str, save_path):
    if encoded_str.startswith("b'") and encoded_str.endswith("'"):
        encoded_str = encoded_str[2:-1]
    decoding_bytes = base64.b64decode(encoded_str)
    folder_path = os.path.dirname(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(save_path, 'wb') as f:
        f.write(decoding_bytes)
# from aiohttp import web
# import aiohttp_cors
def merge_video_audio(video_path, audio_path, output_path):
    # 使用 input() 来加载视频和音频文件
    video_input = ffmpeg.input(video_path)
    audio_input = ffmpeg.input(audio_path)

    # 使用 output() 来合成并保存输出
    ffmpeg.output(video_input, audio_input, output_path, vcodec='copy', acodec='aac', strict='experimental', y='-y').run()

class TFGgenerator():
    def __init__(self):
        # ipdb.set_trace()
        # self.imagequeue = Queue()
        # self.audioqueue = Queue()
        is_init = True
        self.width = 256
        self.height = 256
        # sc = StreamerConfig()
        # sc.source_width = self.width
        # sc.source_height = self.height
        # sc.stream_width = self.width
        # sc.stream_height = self.height
        
        # rtmp_server = "rtmp://localhost:1935/live/livestream"
        # self.fps = 25
        # self.chunk = 320
        # self.sample_rate = 16000
        # sc.stream_fps = self.fps
        # sc.stream_bitrate = 4000000
        # sc.stream_profile = 'main'
        # sc.audio_channel = 1
        # sc.sample_rate = self.sample_rate
        # sc.stream_server = rtmp_server
        # self.streamer = Streamer()
        # ipdb.set_trace()
        # self.streamer.init(sc)
        # # self.streamer.enable_av_debug_log()
        # self.generate_video=False
        # self.role_name = 'nvbobaoyuan'
        
        # self.lock = threading.Lock()
    
    # def loop_video_generate(self):
    #     print(self.generate_video)
    #     sunwukong = '/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/SadTalker/nospeaker/sunwukong'
    #     nvbobaoyuan = '/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/SadTalker/nospeaker/nvbobaoyuan'
    #     wukong_list = read_images_from_folder(sunwukong)
    #     nvbobaoyuan_list = read_images_from_folder(nvbobaoyuan)
    #     empty_audio = np.zeros(self.chunk, dtype=np.float32)
    #     while True:
    #         with self.lock:
    #             temp_gen = self.generate_video
    #         if temp_gen:
    #             print("generate-video is: "+str(self.generate_video))
    #             try:
    #                 t = time.time()
    #                 frame = self.imagequeue.get(timeout=5)
    #                 self.streamer.stream_frame(frame)
    #                 if not self.audioqueue.empty():
    #                     print(f"audioqueue length is: {self.audioqueue.qsize()}")
    #                     if self.audioqueue.qsize() == 1:
    #                         self.audioqueue.put(empty_audio)
    #                     for _ in range(2):
    #                         audio_frame = self.audioqueue.get()
    #                         self.streamer.stream_frame_audio(audio_frame)
    #                 else:
    #                     for _ in range(2):
    #                         # audio_frame = np.zeros(self.chunk, dtype=np.float32)
    #                         self.streamer.stream_frame_audio(empty_audio)
    #                 delay = 0.04 - (time.time() - t)
    #                 if delay > 0:
    #                     time.sleep(delay)
    #                 print("complete generate one frame")
    #             except Exception as e:
    #                 print('=' * 100)
    #                 time.sleep(0.5)
    #                 continue
    #         else:
    #             print(self.role_name)
    #             if self.role_name == '孙悟空':
    #                 for frame in wukong_list:
    #                     t = time.time()
    #                     self.streamer.stream_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #                     for i in range(2):
    #                         audio_frame = np.zeros(self.chunk, dtype=np.float32)
    #                         self.streamer.stream_frame_audio(audio_frame)
    #                     delay = 0.04 - (time.time() - t)
    #                     if delay > 0:
    #                         time.sleep(delay)
    #             elif self.role_name == 'nvbobaoyuan':
    #                 for frame in nvbobaoyuan_list:
    #                     t = time.time()
    #                     self.streamer.stream_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #                     for i in range(2):
    #                         audio_frame = np.zeros(self.chunk, dtype=np.float32)
    #                         self.streamer.stream_frame_audio(audio_frame)
    #                     delay = 0.04 - (time.time() - t)
    #                     if delay > 0:
    #                         time.sleep(delay)
    #             else:
    #                 num_images = 25
    #                 for _ in range(num_images):
    #                     t = time.time()
    #                     blank_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
    #                     self.streamer.stream_frame(cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB))
    #                     for i in range(2):
    #                         audio_frame = np.zeros(self.chunk, dtype=np.float32)
    #                         self.streamer.stream_frame_audio(audio_frame)
    #                     delay = 0.04 - (time.time() - t)
    #                     if delay > 0:
    #                         time.sleep(delay)

    # def loop_gen(audio_temp_path, face_temp_path):
    #     audio_file_path =  audio_temp_path
    #     chunk = 320  # 320 samples per chunk (20ms * 16000 / 1000)
    #     def create_bytes_stream():
    #         stream, sr = sf.read(audio_file_path)
    #         print(f'[INFO]tts audio stream {sr}: {stream.shape}')
    #         stream = stream.astype(np.float32)

    #         if stream.ndim > 1:
    #             print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
    #             stream = stream[:, 0]

    #         if sr != self.sample_rate and stream.shape[0] > 0:
    #             print(f'[WARN] audio sample rate is {sr}, resampling into {self.sample_rate}.')
    #             stream = resampy.resample(x=stream, sr_orig=sr, sr_new=self.sample_rate)
            
    #         return stream
    #     def push_audio():
    #         wav = create_bytes_stream()
    #         start_idx = 0
    #         while start_idx < wav.shape[0]:
    #             if start_idx + chunk > wav.shape[0]:
    #                 af = wav[start_idx:wav.shape[0]]
    #                 start_idx = 0
    #                 break
    #             else:
    #                 af = wav[start_idx:start_idx + chunk]
    #                 start_idx = start_idx + chunk
    #             self.audioqueue.put(af)
        
    #     push_audio()
    #     # todo: only change l_batch_main for our model
    #     for frame in l_batch_main(audio_temp_path, face_temp_path):
    #         self.imagequeue.put(frame)
    #     while True:
    #         time.sleep(0.01)
    #         with self.lock:
    #             print(self.imagequeue.qsize())
    #             if self.imagequeue.qsize() == 0:
    #                 self.generate_video = False
    #                 break
    #     print("complete!")
    #     time.sleep(2)

    def gen_video(self, audio_temp_path, face_temp_path):
        frames = []
        for frame in l_batch_main(audio_temp_path, face_temp_path):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
        if len(frames) == 0:
            print("没有找到图像文件")
            return
        frame_height, frame_width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 选择编码格式
        video_writer = cv2.VideoWriter('./output_video.mp4', fourcc, 25, (frame_width, frame_height))
        # ipdb.set_trace()
        # todo: ffmpeg overwrite
        """
        File './output.mp4' already exists. Overwrite ? [y/N] Not overwriting - exiting
INFO:     192.168.1.50:54504 - "POST /getv HTTP/1.1" 500 Internal Server Error
ERROR:    Exception in ASGI application"""
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        merge_video_audio('./output_video.mp4', audio_temp_path, './output.mp4')
        
tfggen = TFGgenerator()

class Item(BaseModel):
    text: str
    role: str

class CRole(BaseModel):
    role: str

# face_temp_path = "/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/Wav2Lip/temp/face_temp.png"
audio_temp_path = '/home/tzheng2/workspace/scir-y2/stream-demo/stream-demo/serverfolder/models/fishspeech/fake.wav'

img_path_dict = {
    "可莉": "./assets/klee.png",
    "孙悟空": "./assets/sunwukong.png",
    "猪八戒": "./assets/zhubajie.png",
    "唐僧": "./assets/tangseng.png",
    "nvbobaoyuan": "./assets/nvbobaoyuan.png"
}

@app.post("/changerole")
def changerole(crole: CRole):
    tfggen.role_name = crole.role
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(tfggen.loop_video_generate())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
    
@app.post("/getv")
async def videogen(item: Item):
    global audio_temp_path
    print('enter')
    face_temp_path = img_path_dict.get(item.role)
    data = {
        "text": item.text,
        "role": item.role
    }
    requests.post(url, json=data)
    tfggen.gen_video(audio_temp_path, face_temp_path)
    return FileResponse("./output_video.mp4")

def main():
    face_temp_path = '/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/SadTalker/assert/klee.png'
    tfggen.gen_video(audio_temp_path, face_temp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9960)
    # main()
# appasync = web.Application()
# appasync.router.add_post("/offer", offer)
# cors = aiohttp_cors.setup(appasync, defaults={
#     "*": aiohttp_cors.ResourceOptions(
#         allow_credentials=True,
#         expose_headers="*",
#         allow_headers="*",
#     )
# })
# for route in list(appasync.router.routes()):
#     cors.add(route)
# def run_server(runner):
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(runner.setup())
#     site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
#     print("http://0.0.0.0:8010")
#     loop.run_until_complete(site.start())
#     if opt.transport=='rtcpush':
#         loop.run_until_complete(run(opt.push_url))
#     loop.run_forever()
# Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
# async def main():
#     # 启动异步任务
#     # asyncio.create_task(animate_from_coeff.loop_video_generate())

#     # 启动 uvicorn 服务器
#     config = uvicorn.Config("sadtalker_demo:app", host="0.0.0.0", port=9960, log_level="info")
#     server = uvicorn.Server(config)
#     await server.serve()

# if __name__ == "__main__":
#     asyncio.run(main())