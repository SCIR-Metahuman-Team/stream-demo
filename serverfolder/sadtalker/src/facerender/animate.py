import os
import cv2
import yaml
import numpy as np
import warnings
from skimage import img_as_ubyte
import safetensors
import safetensors.torch 
warnings.filterwarnings('ignore')
# import ffmpeg
import time
import queue
from rtmp_streaming import StreamerConfig, Streamer
import shutil
from queue import Queue
import threading
from threading import Thread,Event
import soundfile as sf
import resampy
import ffmpeg
import imageio
import torch
import torchvision
from PIL import Image

from src.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from src.facerender.modules.mapping import MappingNet
from src.facerender.modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from src.facerender.modules.make_animation import make_animation, make_few_batch_animation

from pydub import AudioSegment 
from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
from src.utils.paste_pic import paste_pic
from src.utils.videoio import save_video_with_watermark
def create_m3u8_by_totalTime(totalTime: int, save_path_name: str, time_gap=1):
    '''根据总时长，按每5s一段，创建一个m3u8文件，返回每个ts文件的名字队列
        :param totalTime 总时长，ms
        :param save_path_name m3u8文件要存储的路径及名称，以.m3u8为后缀
        :returns 返回每个ts的名字的队列对象及最后一个ts的时长(ms)
    '''
    dir = os.path.dirname(save_path_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    segment = int(totalTime / (1000 * time_gap)) if totalTime % (1000 * time_gap) == 0 else int(totalTime / (1000 * time_gap)) + 1
    tsQueue = queue.Queue(segment)
    with open(save_path_name, 'w') as m3u8:
        m3u8.write('#EXTM3U\n')
        m3u8.write('#EXT-X-VERSION:3\n')
        m3u8.write('#EXT-X-MEDIA-SEQUENCE:0\n')  # 当播放打开M3U8时，以这个标签的值作为参考，播放对应的序列号的切片
        m3u8.write('#EXT-X-ALLOW-CACHE:YES\n')
        m3u8.write('#EXT-X-TARGETDURATION:6\n')  # ts播放的最大时长，s
        lastTime = -1
        for i in range(segment):
            if i + 1 == segment:
                lastTime = totalTime % (1000 * time_gap)
            m3u8.write(f'#EXTINF:{time_gap if lastTime < 0 else lastTime / 1000},\n')  # ts时长，注意有个逗号
            m3u8.write(f'{i}.ts\n')
            tsQueue.put(f'{i}.ts')
        m3u8.write('#EXT-X-ENDLIST')
    return tsQueue, lastTime

def seconds_to_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_ts(save_path:str,audio_path_name:str, frame_ndarray:np.ndarray,ts_index:int, time_gap=1):
    '''创建一个time_gap s时长的ts文件'''
    print(frame_ndarray.shape)
    tmp_file = os.path.join(save_path,f'_tmp_quiet_{ts_index}.ts')
    height,width,c = frame_ndarray[0].shape
    print(f'======>视频width:{width},height:{height}')
    #图像写入ts文件
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    process = (
        ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height),framerate=25)
        .output(tmp_file, vcodec='libx264', r=25,output_ts_offset=ts_index * time_gap,hls_time=time_gap,hls_segment_type='mpegts') #r:帧率，output_ts_offset：ts文件序列
        .global_args('-y')  # 覆盖同名文件
        .run_async(pipe_stdin=True)
    )
    for frame in frame_ndarray:
        process.stdin.write(frame.astype(np.uint8).tobytes())
        time.sleep(0.01)
    process.stdin.close()
    process.wait()
    tmp_ts = ffmpeg.input(tmp_file)
    audio = ffmpeg.input(audio_path_name, ss=seconds_to_time(ts_index * time_gap),t=str(frame_ndarray.shape[0] / 25))
    joined = ffmpeg.concat(tmp_ts.video, audio, v=1, a=1).node
    out = ffmpeg.output(joined[0], joined[1],tmp_file.replace('_tmp_quiet_',''),vcodec='libx264', r=25,output_ts_offset=ts_index * time_gap,hls_time=time_gap,hls_segment_type='mpegts').global_args('-y')
    out.run()




try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False
# def seconds_to_time(seconds):
#     hours = seconds // 3600
#     minutes = (seconds % 3600) // 60
#     seconds = seconds % 60
#     return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# def create_ts_with_5sec(save_path:str,frame_ndarray:np.ndarray,ts_index:int, audio_path_name:str):
#     '''创建一个5s时长的ts文件'''
#     print(frame_ndarray.shape)
#     tmp_file = os.path.join(save_path,f'_tmp_quiet_{ts_index}.ts')
#     height,width,c = frame_ndarray[0].shape
#     print(f'======>视频width:{width},height:{height}')
#     #图像写入ts文件
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     process = (
#         ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height),framerate=25)
#         .output(tmp_file, vcodec='libx264', r=25,output_ts_offset=ts_index * 5,hls_time=5,hls_segment_type='mpegts') #r:帧率，output_ts_offset：ts文件序列
#         .global_args('-y')  # 覆盖同名文件
#         .run_async(pipe_stdin=True)
#     )
#     for frame in frame_ndarray:
#         process.stdin.write(frame.astype(np.uint8).tobytes())
#         time.sleep(0.01)
#     process.stdin.close()
#     process.wait()
#     tmp_ts = ffmpeg.input(tmp_file)
#     audio = ffmpeg.input(audio_path_name, ss=seconds_to_time(ts_index * 5),t=str(frame_ndarray.shape[0] / 25))
#     joined = ffmpeg.concat(tmp_ts.video, audio, v=1, a=1).node
#     out = ffmpeg.output(joined[0], joined[1],tmp_file.replace('_tmp_quiet_',''),vcodec='libx264', r=25,output_ts_offset=ts_index * 5,hls_time=5,hls_segment_type='mpegts').global_args('-y')
#     out.run()
def read_images_from_folder(folder_path):
    image_list = []

    # 获取文件夹内的所有文件并排序
    files = sorted([f for f in os.listdir(folder_path) if (f.endswith('.jpg') or f.endswith('.png'))])

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
        # 使用cv2.imread读取图像文件
        image = cv2.imread(file_path)
        
        # 检查是否成功读取图像
        if image is not None:
            image_list.append(image)

    # 将图像列表转换为numpy数组
    image_array = np.array(image_list, dtype=np.uint8)
    return image_array
class AnimateFromCoeff():

    def __init__(self, sadtalker_path, device):

        with open(sadtalker_path['facerender_yaml']) as f:
            config = yaml.safe_load(f)

        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                    **config['model_params']['common_params'])
        kp_extractor = KPDetector(**config['model_params']['kp_detector_params'],
                                    **config['model_params']['common_params'])
        he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
        mapping = MappingNet(**config['model_params']['mapping_params'])

        generator.to(device)
        kp_extractor.to(device)
        he_estimator.to(device)
        mapping.to(device)
        for param in generator.parameters():
            param.requires_grad = False
        for param in kp_extractor.parameters():
            param.requires_grad = False 
        for param in he_estimator.parameters():
            param.requires_grad = False
        for param in mapping.parameters():
            param.requires_grad = False

        if sadtalker_path is not None:
            if 'checkpoint' in sadtalker_path: # use safe tensor
                self.load_cpk_facevid2vid_safetensor(sadtalker_path['checkpoint'], kp_detector=kp_extractor, generator=generator, he_estimator=None)
            else:
                self.load_cpk_facevid2vid(sadtalker_path['free_view_checkpoint'], kp_detector=kp_extractor, generator=generator, he_estimator=he_estimator)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.")

        if  sadtalker_path['mappingnet_checkpoint'] is not None:
            self.load_cpk_mapping(sadtalker_path['mappingnet_checkpoint'], mapping=mapping)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.") 

        self.kp_extractor = kp_extractor
        self.generator = generator
        self.he_estimator = he_estimator
        self.mapping = mapping
        self.kp_extractor.eval()
        self.generator.eval()
        self.he_estimator.eval()
        self.mapping.eval()
        
        self.device = device
        # self.imagequeue = Queue()
        # self.audioqueue = Queue()
        # is_init = True
        # self.width = 256
        # self.height = 256
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
        # self.streamer.init(sc)
        # # self.streamer.enable_av_debug_log()
        # self.generate_video=False
        # self.role_name = 'nvbobaoyuan'
        
        # self.lock = threading.Lock()

    def load_cpk_facevid2vid_safetensor(self, checkpoint_path, generator=None, 
                        kp_detector=None, he_estimator=None,  
                        device="cpu"):

        checkpoint = safetensors.torch.load_file(checkpoint_path)

        if generator is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'generator' in k:
                    x_generator[k.replace('generator.', '')] = v
            generator.load_state_dict(x_generator)
        if kp_detector is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'kp_extractor' in k:
                    x_generator[k.replace('kp_extractor.', '')] = v
            kp_detector.load_state_dict(x_generator)
        if he_estimator is not None:
            x_generator = {}
            for k,v in checkpoint.items():
                if 'he_estimator' in k:
                    x_generator[k.replace('he_estimator.', '')] = v
            he_estimator.load_state_dict(x_generator)
        
        return None

    def load_cpk_facevid2vid(self, checkpoint_path, generator=None, discriminator=None, 
                        kp_detector=None, he_estimator=None, optimizer_generator=None, 
                        optimizer_discriminator=None, optimizer_kp_detector=None, 
                        optimizer_he_estimator=None, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if he_estimator is not None:
            he_estimator.load_state_dict(checkpoint['he_estimator'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_he_estimator is not None:
            optimizer_he_estimator.load_state_dict(checkpoint['optimizer_he_estimator'])

        return checkpoint['epoch']
    
    def load_cpk_mapping(self, checkpoint_path, mapping=None, discriminator=None,
                 optimizer_mapping=None, optimizer_discriminator=None, device='cpu'):
        checkpoint = torch.load(checkpoint_path,  map_location=torch.device(device))
        if mapping is not None:
            mapping.load_state_dict(checkpoint['mapping'])
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if optimizer_mapping is not None:
            optimizer_mapping.load_state_dict(checkpoint['optimizer_mapping'])
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])

        return checkpoint['epoch']

    def generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess='crop', img_size=256):
        source_image=x['source_image'].type(torch.FloatTensor)
        source_semantics=x['source_semantics'].type(torch.FloatTensor)
        target_semantics=x['target_semantics_list'].type(torch.FloatTensor) 
        source_image=source_image.to(self.device)
        source_semantics=source_semantics.to(self.device)
        target_semantics=target_semantics.to(self.device)
        if 'yaw_c_seq' in x:
            yaw_c_seq = x['yaw_c_seq'].type(torch.FloatTensor)
            yaw_c_seq = x['yaw_c_seq'].to(self.device)
        else:
            yaw_c_seq = None
        if 'pitch_c_seq' in x:
            pitch_c_seq = x['pitch_c_seq'].type(torch.FloatTensor)
            pitch_c_seq = x['pitch_c_seq'].to(self.device)
        else:
            pitch_c_seq = None
        if 'roll_c_seq' in x:
            roll_c_seq = x['roll_c_seq'].type(torch.FloatTensor) 
            roll_c_seq = x['roll_c_seq'].to(self.device)
        else:
            roll_c_seq = None

        frame_num = x['frame_num']
        # generate low-pre
        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor, self.he_estimator, self.mapping, 
                                        yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp = True)
        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])
        predictions_video = predictions_video[:frame_num]

        video = []
        for idx in range(predictions_video.shape[0]):
            image = predictions_video[idx]
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            video.append(image)
        result = img_as_ubyte(video)

        ### the generated video is 256x256, so we keep the aspect ratio, 
        original_size = crop_info[0]
        if original_size:
            result = [ cv2.resize(result_i,(img_size, int(img_size * original_size[1]/original_size[0]) )) for result_i in result ]
        
        video_name = x['video_name']  + '.mp4'
        path = os.path.join(video_save_dir, 'temp_'+video_name)
        
        imageio.mimsave(path, result,  fps=float(25))
        return path
        av_path = os.path.join(video_save_dir, video_name)
        return_path = av_path 
        
        audio_path =  x['audio_path'] 
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
        new_audio_path = os.path.join(video_save_dir, audio_name+'.wav')
        start_time = 0
        # cog will not keep the .mp3 filename
        sound = AudioSegment.from_file(audio_path)
        frames = frame_num 
        end_time = start_time + frames*1/25*1000
        word1=sound.set_frame_rate(16000)
        word = word1[start_time:end_time]
        word.export(new_audio_path, format="wav")

        save_video_with_watermark(path, new_audio_path, av_path, watermark= False)
        print(f'The generated video is named {video_save_dir}/{video_name}') 
        return return_path
        if 'full' in preprocess.lower():
            # only add watermark to the full image.
            video_name_full = x['video_name']  + '_full.mp4'
            full_video_path = os.path.join(video_save_dir, video_name_full)
            return_path = full_video_path
            paste_pic(path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop= True if 'ext' in preprocess.lower() else False)
            print(f'The generated video is named {video_save_dir}/{video_name_full}') 
        else:
            full_video_path = av_path 

        #### paste back then enhancers
        if enhancer:
            video_name_enhancer = x['video_name']  + '_enhanced.mp4'
            enhanced_path = os.path.join(video_save_dir, 'temp_'+video_name_enhancer)
            av_path_enhancer = os.path.join(video_save_dir, video_name_enhancer) 
            return_path = av_path_enhancer

            try:
                enhanced_images_gen_with_len = enhancer_generator_with_len(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            except:
                enhanced_images_gen_with_len = enhancer_list(full_video_path, method=enhancer, bg_upsampler=background_enhancer)
                imageio.mimsave(enhanced_path, enhanced_images_gen_with_len, fps=float(25))
            
            save_video_with_watermark(enhanced_path, new_audio_path, av_path_enhancer, watermark= False)
            print(f'The generated video is named {video_save_dir}/{video_name_enhancer}')
            os.remove(enhanced_path)

        os.remove(path)
        os.remove(new_audio_path)

        return return_path

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
                            # time.sleep(delay)
                
    def loop_batch_generate(self, x, video_save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess='crop', img_size=256):
        source_image=x['source_image'].type(torch.FloatTensor)
        source_semantics=x['source_semantics'].type(torch.FloatTensor)
        target_semantics=x['target_semantics_list'].type(torch.FloatTensor) 
        source_image=source_image.to(self.device)
        source_semantics=source_semantics.to(self.device)
        target_semantics=target_semantics.to(self.device)
        if 'yaw_c_seq' in x:
            yaw_c_seq = x['yaw_c_seq'].type(torch.FloatTensor)
            yaw_c_seq = x['yaw_c_seq'].to(self.device)
        else:
            yaw_c_seq = None
        if 'pitch_c_seq' in x:
            pitch_c_seq = x['pitch_c_seq'].type(torch.FloatTensor)
            pitch_c_seq = x['pitch_c_seq'].to(self.device)
        else:
            pitch_c_seq = None
        if 'roll_c_seq' in x:
            roll_c_seq = x['roll_c_seq'].type(torch.FloatTensor) 
            roll_c_seq = x['roll_c_seq'].to(self.device)
        else:
            roll_c_seq = None
        # =====================================  generate stream  ===================================================
        
        # audio_file_path =  x['audio_path']
        # chunk = 320  # 320 samples per chunk (20ms * 16000 / 1000)
        # def create_bytes_stream():
        #     stream, sr = sf.read(audio_file_path)
        #     print(f'[INFO]tts audio stream {sr}: {stream.shape}')
        #     stream = stream.astype(np.float32)

        #     if stream.ndim > 1:
        #         print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
        #         stream = stream[:, 0]

        #     if sr != self.sample_rate and stream.shape[0] > 0:
        #         print(f'[WARN] audio sample rate is {sr}, resampling into {self.sample_rate}.')
        #         stream = resampy.resample(x=stream, sr_orig=sr, sr_new=self.sample_rate)

        #     return stream
        # def push_audio():
        #     wav = create_bytes_stream()
        #     start_idx = 0
        #     while start_idx < wav.shape[0]:
        #         if start_idx + chunk > wav.shape[0]:
        #             af = wav[start_idx:wav.shape[0]]
        #             start_idx = 0
        #             break
        #         else:
        #             af = wav[start_idx:start_idx + chunk]
        #             start_idx = start_idx + chunk
        #         self.audioqueue.put(af)
        
        # push_audio()
        # =====================================  end gen stream  ===================================================
        
        last_predictions = []
        
        count = 0
        for prediction_frame in make_few_batch_animation(source_image, source_semantics, target_semantics, 
                                         self.generator, self.kp_extractor, self.he_estimator, self.mapping, 
                                         yaw_c_seq, pitch_c_seq, roll_c_seq, use_exp = True):
            count += 1
            if count >= target_semantics.shape[1] / 2:
                self.generate_video = True
            last_predictions.append(prediction_frame[1])  # 3,256,256
            image = prediction_frame[0]
            
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            frame = img_as_ubyte(image)  # numpy.ndarray
            original_size = crop_info[0]
            if original_size:
                frame = cv2.resize(frame,(img_size, int(img_size * original_size[1]/original_size[0]) ))
            yield frame
            # img_frame = Image.fromarray(frame)
            # img_frame.save(f'./output_img_{count}.jpg')
            # self.imagequeue.put(frame)
        for image in last_predictions:
            time.sleep(0.01)
            t = time.time()
            image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
            frame = img_as_ubyte(image)  # numpy.ndarray
            original_size = crop_info[0]
            if original_size:
                frame = cv2.resize(frame,(img_size, int(img_size * original_size[1]/original_size[0]) ))
            # img_frame = Image.fromarray(frame)
            # img_frame.save(f'./output_img_{count}.jpg')
            yield frame
            # self.imagequeue.put(frame)
        # while True:
        #     time.sleep(0.01)
        #     with self.lock:
        #         print(self.imagequeue.qsize())
        #         if self.imagequeue.qsize() == 0:
        #             self.generate_video = False
        #             break
        # print("complete!")
        # time.sleep(2)