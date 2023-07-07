# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread

import cv2
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter

from .mmaction.apis1 import init_recognizer
from .mmaction.datasets1.pipelines import Compose

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL  #设置字体的
FONTSCALE = 1  #字体大小
FONTCOLOR = (255, 255, 255)  # BGR, white  #颜色白色
MSGCOLOR = (128, 128, 128)  # BGR, gray   
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]  


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    #创建配置文件
    parser.add_argument('--config',default='mmaction2/configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py', help='test config file path')
    #本地配置文件的地址
    parser.add_argument('--checkpoint',default='mmaction2/checkpoints/tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219-032c8e94.pth', help='checkpoint file')
    #权重文件
    parser.add_argument('--label',default='mmaction2/tools/data/kinetics/label_map_k400.txt', help='label file')
    #标签文件
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
        #用的CPU，CUDA
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
        #设备id
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.02,
        help='recognition score threshold')
        #识别分数的阈值
    parser.add_argument(
        '--average-size',
        type=int,
        default=5,
        help='number of latest clips to be averaged for prediction')
        #要为预测的平均剪辑数
    parser.add_argument(
        '--drawing-fps',
        type=int,
        default=20,
        help='Set upper bound FPS value of the output drawing')
        #设置输出图形的上限fps值 
    parser.add_argument(
        '--inference-fps',
        type=int,
        default=4,
        help='Set upper bound FPS value of model inference')
        #模型推理的上限FPS
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    assert args.drawing_fps >= 0 and args.inference_fps >= 0, \
        'upper bound FPS value of drawing and inference should be set as ' \
        'positive number, or zero for no limit'
    return args


def show_results():
    print('Press "Esc", "q" or "Q" to exit')

    text_info = {}  #创建一个列表
    cur_time = time.time()  #记录现在时间
    n=0
    while True:  #循环运行
        msg = 'Waiting for action ...'
        _, frame = camera.read()
        #先读取摄像头
        #print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz')
        #print(frame)
        #print('-1-1-1-1-1-1-1-1-1--1-1-1-1-1-1--1')
        #print(frame[:, :, ::-1])
        frame_queue.append(np.array(frame[:, :, ::-1]))
        #n=n+1
        #if len(frame_queue) < sample_length :
            #print('show函数 读帧 ：'+str(n))
            #mytime2=time.time()
            #print(mytime1-mytime2)
        #elif len(frame_queue) == sample_length :
            #print('队列满了')
            #print(frame_queue[-1])
        #读取帧传入队列
        if len(result_queue) != 0:
        #结果队列如果不等于0，结果帧有东西
            text_info = {}  #文本信息变成空字典。
            results = result_queue.popleft()
            #结果帧的东西弹出放到results 里面
            for i, result in enumerate(results):
                selected_label, score = result
                if score < threshold: #小于阈值就不显示
                    break
                location = (0, 40 + i * 20) #放的位置
                text = selected_label + ': ' + str(round(score, 2))
                text_info[location] = text #字典加入location，然后放入分数
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)  #放上去

        elif len(text_info) != 0:
        #结果帧等于0，有文本信息，就是平常每帧没有显示的时候
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        else:
        #啥都没有的情况下
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)
                        #把文字放在帧上面
                        #文字是msg，一开始的时候这个是waiting
                        
	#上面if做完之后显示画面
        cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)
        #cv2.waitKey(1) 1为参数，单位毫秒，表示间隔时间
        #waitKey(int delay)键盘绑定函数，共一个参数，表示等待毫秒数，将等待特定的几毫秒，看键盘是否有输入，
        #如果delay大于0，那么超过delayms后，如果没有按键，那么会返回-1，
        #如果按键那么会返回键盘值,返回值为ASCII值。如果其参数为0，则表示无限期的等待键盘输入。


        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break


        #这个没懂，这个终止计时是用的不知道啥
        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()


def inference():
    score_cache = deque()  #分数缓存创建一个双向队列
    scores_sum = 0 #分数和
    cur_time = time.time() #计时
    while True:
        cur_windows = []  #创建窗口

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:  #队列长度等于采样长度就是说队列满了
                cur_windows = list(np.array(frame_queue)) #当前窗口就等于这个队列中的每一个帧
                #print('dddddddddddddddddddddddddddddddddddddddd')
                #print(data)
                if data['img_shape'] is None:
                    #print('iiiiiiiiiiiiiiiiiiimmmmmmmmmmmmmmmmmm')
                    #print(frame_queue.popleft().shape[:2])
                    data['img_shape'] = frame_queue.popleft().shape[:2]
                    #这里动态的设置了图片大小根据你图片输入的大小来设置。

        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        #拿到的数据是这个窗口的输入数据
        #print('cur11111111111')
        #print(cur_data)
        #上面是拿到了数据，下面这一部就是对数据做了一个流水线的处理
        cur_data = test_pipeline(cur_data)
        #出来的数据就变成tenor了
        #print('cur2222222222222')
        #print(cur_data)
        cur_data = collate([cur_data], samples_per_gpu=1)
        #print('cur333333333333')
        #print(cur_data)
        if next(model.parameters()).is_cuda:
            cur_data = scatter(cur_data, [device])[0]
        #print('cur444444444444')
        #print(cur_data)

        with torch.no_grad():
            scores = model(return_loss=False, **cur_data)[0]
        print('scores :+++++++++++++')
        #print(scores)
        score_cache.append(scores)
        #加入缓存
        scores_sum += scores
        #分数求和
        #print(len(score_cache))
        #print('+++++++++++++')
        #print(average_size)
        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            #处以平均数
            num_selected_labels = min(len(label), 5)
            #选择标签数，默认为5，不到5个的用所有标签

            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]

            result_queue.append(results)
            #结果队列里面存放数据
            scores_sum -= score_cache.popleft()
            #队列左侧出去，右侧进来

        if inference_fps > 0:
            # add a limiter for actual inference fps <= inference_fps
            sleep_time = 1 / inference_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()

    camera.release()
    cv2.destroyAllWindows()


def main():
    global frame_queue, camera, frame, results, threshold, sample_length, \
        data, test_pipeline, model, device, average_size, label, \
        result_queue, drawing_fps, inference_fps, mytime1, mytime2
        
        #mytime是我自己测试用于打印测试的变量
        
        
#一堆全局变量
#1.frame_queue 帧序列（队列）
#2.camera
#3.frame
#4.results
#5.threshold  阈值
#6.sample_length 采样长度
#7.data
#8.test_pipeline 测试流水线
#9.model
#10.device
#11.average_size
#12.label
#13.result_queue 结果队列
#14.drawing_fps
#15.inference_fps
#

    mytime1=time.time()
    args = parse_args() #创建参数接受器
    average_size = args.average_size
    threshold = args.threshold
    drawing_fps = args.drawing_fps
    inference_fps = args.inference_fps

    device = torch.device(args.device)
    
    #赋值这波变量

    cfg = Config.fromfile(args.config)
    #从文件里面读取配置文件
    print('CFG')
    print(cfg)
    cfg.merge_from_dict(args.cfg_options)
    #可选的配置文件放进来

    model = init_recognizer(cfg, args.checkpoint, device=device)
    #用配置文件，权重文件，设备初始化模型
    camera = cv2.VideoCapture(args.camera_id)
    #拿到camera
    data = dict(img_shape=None, modality='RGB', label=-1)
    #数据是字典形式，图片大小这个参数是没有的，modality='RGB'，label是-1
    print('label :++++++++++++++')
    print(args.label)
    #这里用的标注数据是本地的一个txt
    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    
    cfg = model.cfg
    print('model cfg :++++++++')
    print(cfg)
    sample_length = 0
    #这里把采样长度竟然设置成了0
    pipeline = cfg.data.test.pipeline
    print('pipeline :+++++++++++++++++')
    print(pipeline)
    #这里pipeline参数要注意一下，用的是opencvinit初始化，开一个线程，
    #用的是抽帧的方法，片段长度为1，帧间干涉1，片段数是25，测试模式。
    #decode用的opencv decode，然后是一些数据预处理的方法。
    pipeline_ = pipeline.copy()
    print('data :++++++++++')
    print(data)
    for step in pipeline:
        if 'SampleFrames' in step['type']:
        #调整采样帧时候的参数，采样长度改了变成了片段长度*片段数
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
            
    #print('0000000000000000')
    #print(data)
    #data多了两个参数
    #print('0000000000000000')
    #print(pipeline)
    #print(pipeline_)
    test_pipeline = Compose(pipeline_)
    #print('test_pipeline :')
    #print(test_pipeline) #只是变成了compose形式的

    assert sample_length > 0  #改过了采样长度了

    try:
        frame_queue = deque(maxlen=sample_length) #创建队列，python的双向队列，最大长度是采样长度
        result_queue = deque(maxlen=1) #结果队列，队列长度就一
        #要注意这两个队列都是全局变量
        pw = Thread(target=show_results, args=(), daemon=True)  #启动线程，一个线程启动展示结果函数
        pr = Thread(target=inference, args=(), daemon=True) #这个线程用于推理
        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
