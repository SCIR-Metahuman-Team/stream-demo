from glob import glob
import shutil
import torch
from time import  strftime
import os, sys, time
import uvicorn
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
import json
from threading import Thread,Event
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from contextlib import asynccontextmanager
import base64
from pydantic import BaseModel
import requests
import asyncio
url = 'http://127.0.0.1:9999/fish'

def decode_base64_str(encoded_str, save_path):
    if encoded_str.startswith("b'") and encoded_str.endswith("'"):
        encoded_str = encoded_str[2:-1]
    decoding_bytes = base64.b64decode(encoded_str)
    folder_path = os.path.dirname(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(save_path, 'wb') as f:
        f.write(decoding_bytes)

@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

result_dir = './results'
pose_style=0
batch_size=2
size=256
expression_scale=1
input_yaw = None
input_pitch = None
input_rollt = None
enhancer='gfpgan'
background_enhancer = None
cpu = False
face3dvis=False
still = False
preprocess='crop'
verbose = False
old_version = False
net_recon='resnet50'
use_last_fc = False
bfm_folder = './checkpoints/BFM_Fitting/'
bfm_model = 'BFM_model_front.mat'
focal=1015
center=112
camera_d=10
z_near=5
z_far=15


if torch.cuda.is_available() and not cpu:
    device = "cuda"
else:
    device = "cpu"


save_dir = os.path.join(result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
os.makedirs(save_dir, exist_ok=True)
pose_style = pose_style
device = device
batch_size = batch_size
input_yaw_list = input_yaw
input_pitch_list = input_pitch
input_roll_list = None
ref_eyeblink = None
ref_pose = None
checkpoint_dir = './checkpoints'
current_root_path = ''
sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, old_version, preprocess)

#init model
preprocess_model = CropAndExtract(sadtalker_paths, device)

audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)

animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)


def l_batch_main(audio_path, pic_path):
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, preprocess,\
                                                                             source_image_flag=True, pic_size=size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
    if face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=expression_scale, still_mode=still, preprocess=preprocess, size=size)
    for frame in animate_from_coeff.loop_batch_generate(data, save_dir, pic_path, crop_info, \
                                enhancer=enhancer, background_enhancer=background_enhancer, preprocess=preprocess, img_size=size):
        yield frame
    # shutil.move(result, save_dir+'.mp4')
    # print('The generated video is named:', save_dir+'.mp4')
    # if not verbose:
    #     shutil.rmtree(save_dir)
    # return save_dir + '.mp4'


def main(audio_path, pic_path):
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, preprocess,\
                                                                             source_image_flag=True, pic_size=size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
    if face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=expression_scale, still_mode=still, preprocess=preprocess, size=size)
    # import ipdb
    # ipdb.set_trace()
    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=enhancer, background_enhancer=background_enhancer, preprocess=preprocess, img_size=size)
    
    shutil.move(result, save_dir+'.mp4')
    print('The generated video is named:', save_dir+'.mp4')
    if not verbose:
        shutil.rmtree(save_dir)
    return save_dir + '.mp4'


# from aiohttp import web
# import aiohttp_cors

# class Item(BaseModel):
#     text: str
#     role: str

# class CRole(BaseModel):
#     role: str

# # face_temp_path = "/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/Wav2Lip/temp/face_temp.png"
# audio_temp_path = '/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/fish-speech/fish-speech-main/fake.wav'

# img_path_dict = {
#     "可莉": "/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/SadTalker/assert/klee.png",
#     "孙悟空": "/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/SadTalker/assert/sunwukong.png",
#     "猪八戒": "/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/SadTalker/assert/zhubajie.png",
#     "唐僧": "/home/tzheng2/workspace/scir-y1/imgtxt2facialvideo/related_works/SadTalker/assert/tangseng.png"
# }

# @app.post("/changerole")
# def changerole(crole: CRole):
#     animate_from_coeff.role_name = crole.role
#     loop = asyncio.get_event_loop()
#     try:
#         loop.run_until_complete(animate_from_coeff.loop_video_generate())
#     except KeyboardInterrupt:
#         pass
#     finally:
#         loop.close()
    
# @app.post("/getv")
# def videogen(item: Item):
#     global audio_temp_path
#     print('enter')
#     face_temp_path = img_path_dict.get(item.role)
#     data = {
#         "text": item.text,
#         "role": item.role
#     }
#     requests.post(url, json=data)
#     l_batch_main(audio_temp_path, face_temp_path)


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
#     config = uvicorn.Config("sadtalker_demo:app", host="0.0.0.0", port=9980, log_level="info")
#     server = uvicorn.Server(config)
#     await server.serve()

# if __name__ == "__main__":
#     asyncio.run(main())