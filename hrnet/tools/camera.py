from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm
import copy
import cv2
import matplotlib.pyplot as plt
import _init_paths
import models
from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.vis import save_debug_images
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size

torch.multiprocessing.set_sharing_strategy('file_system')  #多进程包

def parse_args():  #处理参数的方法
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args

def main():  #主函数


    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)
#args是参数，就是传进去的文件


    #cfg是日志
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting GPU设置不管
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


#定义模型，不训练
    model = eval('models1.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    #获取摄像头
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    #设置摄像头大小


    dump_input = torch.rand(
        (1, 3, 640, 480)
    )
    # 返回一个张良

    logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))
    #模型综述就一开始打印出来那个

    if cfg.FP16.ENABLED:
        model = network_to_half(model)
        #应该也是用来优化的代码

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))

        #也是搞日志信息的，还读取了模型的位置信息


        model.load_state_dict(torch.load(model_state_file))
        #加载模型参数

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()  #多CPU训练
    model.eval() #模型启动

    # data_loader, test_dataset = make_test_dataloader(cfg)
    #这句是原始的加载数据的语句，他把数据用dataset和dataloader包裹了，这里我们就输入最原始的一帧画面不用这些


#下面先不管，不知道为什么出现了堆叠沙漏网络
    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        #反正应该是将数据归一化方便模型的预测

#解析器
    parser = HeatmapParser(cfg)  #转group，创建一个热图解析器的实例
    all_preds = [] # 预测点
    all_scores = [] #预测点分数

    while True:
        ret, img = cap.read()
        image=img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.numpy()


    # pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None
    # for i, (images, annos) in enumerate(data_loader):
    #     assert 1 == images.size(0), 'Test batch size should be 1'
    #
    #     image = images[0].cpu().numpy()
    #     # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(  #转tranforms
                image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
            )
        #把现在的图片变成不同尺度大小的图片？？？

        with torch.no_grad():   #不进行求导运算的操作，在训练的代码里是没有这步的
            final_heatmaps = None   #先创建一个空的热图
            tags_list = []         #空标签的列表
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):   #这应该是最后一层输出选择分辨率特征的吧
                input_size = cfg.DATASET.INPUT_SIZE   #输入大小
                image_resized, center, scale = resize_align_multi_scale(  #对原图进行变换
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                #同样应该也是变换

                outputs, heatmaps, tags = get_multi_stage_outputs(  #获取不同阶段的输出，转inference
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                # l1=outputs.cpu().numpy()
                # l2=heatmaps.cpu().numpy()
                # l3=tags.cpu().numpy()

                # print('outputs:')
                # print(outputs)
                # print('heatmaps: ')
                # print(heatmaps)
                # print('tags: ')
                # print(tags)

                final_heatmaps, tags_list = aggregate_results(   #最后的热图和标签列表
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            #最后的热图，和标签集
            # print('final_heatmaps:')
            # print(final_heatmaps.shape)   打印可知heatmaps是【1，17，512，704】
            #
            # print('tags: ')
            # print(tags.shape)    tags是[1,17,512,704,2]   这里的1，是batchsize  ，17就是17个点

            grouped, scores = parser.parse(   #解析得到分组，和scores
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            final_results = get_final_preds(   #得到最后的预测
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )
        #
        # print('final_results ')
        # print(final_results)


        # prefix = '{}_{}'.format(os.path.join(final_output_dir, 'result_valid'), i)
            # logger.info('=> write {}'.format(prefix))
        img=save_valid_image(image,final_results,'xxx')
        canvas = copy.deepcopy(img)

        # save_valid_image(image, final_results, '{}.jpg'.format(prefix), dataset=test_dataset.name)
        #     # save_debug_images(cfg, image_resized, None, None, outputs, prefix)

        all_preds.append(final_results)
        all_scores.append(scores)
        cv2.imshow('demo', canvas)  # 一个窗口用以显示原视频
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # if cfg.TEST.LOG_PROGRESS:
    #     pbar.close()
    #
    # name_values, _ = test_dataset.evaluate(
    #     cfg, all_preds, all_scores, final_output_dir
    # )




if __name__ == '__main__':
    main()