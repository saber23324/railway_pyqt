import os.path
import sys
import time
import atexit
import json
# from mmdet.apis import init_detector, inference_detector
# import mmcv
# from mmengine.visualization import Visualizer
# from mmdet.visualization.local_visualizer import DetLocalVisualizer

# from mmdet.structures import DetDataSample

from mmdet.apis import DetInferencer

'''
静态推理脚本，选择模型后按开始推理即可（还没写选择模型按钮），主程序依次给脚本传入推理图片路径或文件夹。
'''
'''

config E:\3_Entrepreneurship\XZT\ShangYiDemo\ShangYiDemo\src\setting\model_configs\custom_config_yolov3_d53_8xb8-ms-608-273e_coco.py
checkpoint E:\3_Entrepreneurship\XZT\ShangYiDemo\ShangYiDemo\src\setting\model_weights\epoch_119.pth
--show-dir ..\E:\3_Entrepreneurship\XZT\ShangYiDemo\ShangYiDemo\src\result_imgs 

'''

config_file_path1 = "model_path/yolov3/yolov3.py"
model_weights_path1 = "model_path/yolo/epoch_119.pth"
config_file_path2 = "model_path/diffusionDet/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py"
model_weights_path2 = "model_path/diffusionDet/iter_45000.pth"
config_file_path3 = "model_path/ViTDet/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py"
model_weights_path3 = "model_path/ViTDet/iter_165000.pth"
# 根据配置文件和 checkpoint 文件构建模型
# model = init_detector(config_file_path2, model_weights_path2, device='cuda:0')
inferencer = DetInferencer(model=config_file_path2, weights=model_weights_path2)

def load_mmdet_model(model_name='Yolov3'):
    if(model_name=='Yolov3'):
        inferencer = DetInferencer(model=config_file_path1, weights=model_weights_path1)
        # model = init_detector(config_file_path1, model_weights_path1, device='cpu')
    elif(model_name=='DiffusionDet'):
        # model = init_detector(config_file_path2, model_weights_path2, device='cpu')
        inferencer = DetInferencer(model=config_file_path2, weights=model_weights_path2)

    elif(model_name=='ViTDet'):
        # model = init_detector(config_file_path3, model_weights_path3, device='cpu')
        inferencer = DetInferencer(model=config_file_path3, weights=model_weights_path3)

    else:
        print("no module")
    
    # 测试单张图片并展示结果
    # img = r'E:\3_Entrepreneurship\XZT\Dataset\three_diseases\JPEGImages\000001.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
# result = inferencer('demo/demo.jpg', return_datasamples=True)
# pprint(result, max_length=4)
# def save():  #保存json文件
#     A = dict()
#     listbigoption = []
#     for bbox in self.graphicsView.bboxPointList:
#         listobject = dict()
#         listxy = bbox[:-1]
#         label = bbox[-1]
#         listobject['points'] = listxy
#         listobject['label'] = str(label)
#         listbigoption.append(listobject)

#     A['shapes'] = listbigoption
#     fileObject = self.curr_pic.split('/')
#     filename = fileObject[-1]
#     A['imagePath'] = str(filename)
#     A['imageData'] = self.base64encode_img(self.curr_pic)
#     name = os.path.splitext(filename)[0]
#     filepath, type = QFileDialog.getSaveFileName(None, '文件保存', name, 'json(*.json)')
#     with open(filepath, 'w', encoding='utf-8') as file_obj:
#         json.dump(A, file_obj, indent=4, ensure_ascii=False)

def read_images_and_inference(folder):
    for filename in os.listdir(folder):
        # 拼接文件的完整路径
        img_path = os.path.join(folder, filename)
        # 检查文件是否是一个文件
        if os.path.isfile(img_path):
            # 检查文件是否是一个图片文件
            # 这里只检查了.jpg和.png，你可以根据需要添加更多的图片格式
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                load_mmdet_imge(img_path)


classes =["KJ","CZ","BS"]
def load_mmdet_imge(img_path):
    try:
        # result=inferencer(img_path, out_dir='detect_results/', no_save_pred=False)
        result=inferencer(img_path)
        num=0
        for sorc in result['predictions'][0]['scores']:
            if(sorc<0.05):
                break
            num +=1
        del(result['predictions'][0]['labels'][num:])
        del(result['predictions'][0]['bboxes'][num:])
        del(result['predictions'][0]['scores'][num:])
        inferdata={"shapes":[],"imagePath":img_path.split('/')[-1]}
        for idx in range(num):
            # result['predictions'][0]['labels'][idx]
                # 125.76903076171875,
                # 134.93624267578124,
                # 191.1462646484375,
                # 188.510400390625
            bboxeslist=[x/1 for x in result['predictions'][0]['bboxes'][idx] ]
            inferdata["shapes"].append({"points":bboxeslist,"label":classes[result['predictions'][0]['labels'][idx]],"scores":result['predictions'][0]['scores'][idx]})
            # inferdata["shapes"].append(result['predictions'][0]['bboxes'][idx])
            # inferdata["shapes"].append(result['predictions'][0]['scores'][idx])
        # print(img_path.split('/')[0:-2])
        
        with open(img_path.split('.')[0]+'.json', 'w', encoding='utf-8') as file_obj:
            json.dump(inferdata, file_obj, indent=4, ensure_ascii=False)


    except Exception as e:
        # print(e)
        with open('interlog.txt','a') as file:
            file.write(str(e)+'\n')
        return
        
    # # 获取可视化器
    # det_visualizer = DetLocalVisualizer()
    # # 获取原图
    # img = mmcv.imread(img_path)
    # img = mmcv.imconvert(img, 'bgr', 'rgb')
    # result_img_name = os.path.basename(img_path)
    # result_img_path = os.path.join('./detect_results',result_img_name)
    # # inferencer('demo/demo.jpg', out_dir='outputs/', no_save_pred=False)

    # # 绘制
    # det_visualizer.add_datasample(
    #     'image',
    #     img,
    #     draw_gt=False,
    #     data_sample=result,
    #     show=False,
    #     wait_time=0,
    #     pred_score_thr=0.5,
    #     out_file=result_img_path)
    # return result_img_path


def cleanup():
    print("Performing cleanup tasks...")


if __name__ == '__main__':
    atexit.register(cleanup)
    while True:
        gettext=sys.stdin
        if(gettext==None):
            time.sleep(0.3)
        else:
            watchdog=0
        for line in gettext:
            with open('interlog.txt','a') as file:
                file.write(line.strip()+"\n")
            result=line.strip().split()
            # print(result[0])
            if(result[0]=="-m"):
                load_mmdet_model(result[1])
            elif(result[0]=="-p"):
                print(load_mmdet_imge(result[1]))
            elif(result[0]=="-d"):
                read_images_and_inference(result[1])
            elif(result[0]=="-q"):
                print('procese quit!')
                sys.exit()
            # split
        # print(sys.argv[1],sys.argv[2])
      
        
# 在一个新的窗口中将结果可视化
# model.show_result(img, result)
# # 或者将可视化结果保存为图片
# model.show_result(img, result, out_file='result.jpg')

# 测试视频并展示结果
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)

