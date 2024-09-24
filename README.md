# stream-demo
## 基本介绍
stream-demo项目主要用于整合所有metahuman相关的输入输出，以一个统一的方式控制前后端之间的通信，并采用rtmp方式进行视频流实时传输。
## 特点
1. 支持自定义数字人
2. 既能够以社交软件般发送文本与图像，又能够直接进行对话，此时会打开摄像头传递说话人的多模态信息
3. 采用基于rtmp方法的流式视频输出，降低输出延迟
4. 支持自定义播放视频
## 项目文件结构
```bash
|stream-demo/           # 项目文件夹
----|clientfolder/      # 前端展示界面相关文件夹
--------|SRS_files/     # SRS服务相关的js，css文件。Todo：后续可能优化
            ......
--------|srs.html       # srs服务的前端网页界面，将chatdemo.py生成的界面内嵌到了该html中。
--------|chatdemo.py    # 基于gradio的聊天交互界面
----|serverfolder/      # 服务器相关文件夹
--------|chatserver.py  # 数据流转发中间服务器
--------|models/        # 模型文件夹，内有多个不同功能的模型项目，将SCIR-MetaHuman-Team中的模型导入到这里
            ......
```
该项目中主要给出<code>clientfolder</code>中所有文件的核心代码，以及<code>serverfolder/chatserver.py</code>核心代码，<code>serverfolder/models/</code>中简单给出了如何进行输出以匹配到chatserver.py中的接收端，从而将内容发送到客户端中。
## 环境配置
后续给出
python中的rtmp流式传输环境搭建，请参考[python-rtmpstream](https://github.com/lipku/python_rtmpstream)
