今天，带大家利用RT-DETR（我们可以换成任意一个模型）+Flask来实现一个目标检测平台小案例，其实现效果如下：

[video(video-ZFR3i85x-1721034971285)(type-bilibili)(url-https://player.bilibili.com/player.html?aid=1156246735)(image-https://img-blog.csdnimg.cn/img_convert/b8ba76f2d9e3a7c43674349d0427d499.jpeg)(title-目标检测案例)]

这个案例很简单，就是让我们上传一张图像，随后选择一下置信度，即可检测出图像中的目标，那么具体该如何实现呢？

## RT-DETR模型推理
在先前的学习过程中，博主对RT-DETR进行来了简要的介绍，作为百度提出的实时性目标检测模型，其无论是速度还是精度均取得了较为理想的效果，今天则主要介绍一下RT-DETR的推理过程，与先前使用`DETR`中使用`pth`权重与网络结构相结合的推理方式不同，RT-DETR中使用的是onnx这种权重文件，因此，我们需要先对onnx文件进行一个简单了解：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f1f646bb779b4d9480815dba9a0e6d32.png)

### ONNX模型文件

```python
import onnx
# 加载模型
model = onnx.load('onnx_model.onnx')
# 检查模型格式是否完整及正确
onnx.checker.check_model(model)
# 获取输出层，包含层名称、维度信息
output = self.model.graph.output
print(output)
```
在原本的DETR类目标检测算法中，推理是采用权重文件与模型结构代码相结合的方式，而在RT-DETR中，则采用onnx模型文件来进行推理，即只需要该模型文件即可。

首先是将pth文件与模型结构进行匹配，从而导出onnx模型文件

```python
"""by lyuwenyu
"""

import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import numpy as np 

from src.core import YAMLConfig

import torch
import torch.nn as nn 


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            print(self.postprocessor.deploy_mode)
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)
    

    model = Model()

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    data = torch.rand(1, 3, 640, 640)
    size = torch.tensor([[640, 640]])

    torch.onnx.export(
        model, 
        (data, size), 
        args.file_name,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=16, 
        verbose=False
    )


    if args.check:
        import onnx
        onnx_model = onnx.load(args.file_name)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')


    if args.simplify:
        import onnxsim
        dynamic = True 
        input_shapes = {'images': data.shape, 'orig_target_sizes': size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(args.file_name, input_shapes=input_shapes, dynamic_input_shape=dynamic)
        onnx.save(onnx_model_simplify, args.file_name)
        print(f'Simplify onnx model {check}...')
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',  default="D:\graduate\programs\RT-DETR-main\RT-DETR-main//rtdetr_pytorch\configs/rtdetr/rtdetr_r18vd_6x_coco.yml",type=str, )
    parser.add_argument('--resume', '-r', default="D:\graduate\programs\RT-DETR-main\RT-DETR-main/rtdetr_pytorch/tools\output/rtdetr_r18vd_6x_coco\checkpoint0024.pth",type=str, )
    parser.add_argument('--file-name', '-f', type=str, default='model.onnx')
    parser.add_argument('--check',  action='store_true', default=False,)
    parser.add_argument('--simplify',  action='store_true', default=False,)

    args = parser.parse_args()

    main(args)

```
随后，便是利用onnx模型文件进行目标检测推理过程了
onnx也有自己的一套流程：
### onnx前向InferenceSession的使用
关于onnx的前向推理，onnx使用了onnxruntime计算引擎。
onnx runtime是一个用于onnx模型的推理引擎。微软联合Facebook等在2017年搞了个深度学习以及机器学习模型的格式标准–ONNX，顺路提供了一个专门用于ONNX模型推理的引擎（onnxruntime）。

```python
import onnxruntime
# 创建一个InferenceSession的实例，并将模型的地址传递给该实例
sess = onnxruntime.InferenceSession('onnxmodel.onnx')
# 调用实例sess的润方法进行推理
outputs = sess.run(output_layers_name, {input_layers_name: x})
```

### 推理详细代码

推理代码如下：
```python
import torch
import onnxruntime as ort
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor

if __name__ == "__main__":
    ##################
    classes = ['car','truck',"bus"]
    ##################
    # print(onnx.helper.printable_graph(mm.graph))
    #############
    img_path = "1.jpg"
    #############
    im = Image.open(img_path).convert('RGB')
    im = im.resize((640, 640))
    im_data = ToTensor()(im)[None]
    print(im_data.shape)

    size = torch.tensor([[640, 640]])
    sess = ort.InferenceSession("model.onnx")
    import time
    start = time.time()
    output = sess.run(
        output_names=['labels', 'boxes', 'scores'],
        #output_names=None,
        input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}
    )
    end = time.time()
    fps = 1.0 / (end - start)
    print(fps)
    # print(type(output))
    # print([out.shape for out in output])

    labels, boxes, scores = output

    draw = ImageDraw.Draw(im)
    thrh = 0.6

    for i in range(im_data.shape[0]):

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        print(i, sum(scr > thrh))
        #print(lab)
        print(f'box:{box}')
        for l, b in zip(lab, box):
            draw.rectangle(list(b), outline='red',)
            print(l.item())

            draw.text((b[0], b[1] - 10), text=str(classes[l.item()]), fill='blue', )
    #############
    im.save('2.jpg')
    #############
```

## 前端代码
前端代码包含两部分，一个是上传页面，一个是显示页面

上传页面如下：
```python
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <title></title>
    <script src="http://www.jq22.com/jquery/jquery-1.10.2.js"></script>
    <style>
        #addCommodityIndex {
            text-align: center;
            width: 300px;
            height: 340px;
            position: absolute;
            left: 50%;
            top: 50%;
            margin: -200px 0 0 -200px;
            border: solid #ccc 1px;
            padding: 35px;
        }
        
        #imghead {
            cursor: pointer;
        }

        .btn {
            width: 100%;
            height: 40px;
            text-align: center;
        }
    </style>
    <link rel="stylesheet" href="../static/css/bootstrap.min.css"  crossorigin="anonymous">
</head>

<body>

    <div id="addCommodityIndex">
        <h2>目标检测</h2>
        <div class="form-group row">
            <form id="upload"  action="/upload" enctype="multipart/form-data" method="POST">
                <img src="">
                <div class="form-group row">
                    <label>上传图像</label>
                    <input type="file" class="form-control"  name='file'>
                </div>
                <div class="form-group row">
                    <label>选择置信度</label>
                    <select class="form-control" name="score" id="exampleFormControlSelect1">
                        <option value="0.5">0.5</option>
                        <option value="0.6">0.6</option>
                        <option value="0.7">0.7</option>
                        <option value="0.8">0.8</option>
                        <option value="0.9">0.9</option>
                    </select>
                </div>
                <div class="form-group row">
                <div class="btn"><input type="submit" class="btn btn-success" value="提交图像" /></div>
                </div>
            </form>
        </div>
    </div>

</body>
</html>
```
显示页面：

```python
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <title></title>
    <script src="http://www.jq22.com/jquery/jquery-1.10.2.js"></script>
    <style>
        #addCommodityIndex {
            text-align: center;
            position: absolute;
            left: 40%;
            top: 50%;
            margin: -200px 0 0 -200px;
            border: solid #ccc 1px;
        }
        #imghead {
            cursor: pointer;
        }
        .result {
            width: 100%;
            height: 100%;
            text-align: center;
        }
    </style>
    <link rel="stylesheet" href="../static/css/bootstrap.min.css"  crossorigin="anonymous">
</head>

<body>

<div id="addCommodityIndex">
<div class="card mb-3" style="max-width: 680px;">
    <div class="row no-gutters">
        <div class="col-md-5">
            <img src="../static/img/result.jpg" class="result">
        </div>
        <div class="col-md-5">
            <div class="card-body">
                <h5 class="card-title">检测结果</h5>
                <p class="card-text">目标数量：{{num}}</p>
                <p class="card-text">检测速度：{{fps}} 帧/秒</p>
                <a  href="/home" class="btn btn-success">继续提交</a>
            </div>
        </div>
    </div>
</div>
</div>
</body>
</html>
```
Flask框架代码：

```python
# -*- coding: utf-8 -*-
from flask import Flask,request,render_template
import json
import os
import time
app = Flask(__name__)
import infer
@app.route('/home',methods=['GET'])
def home():
    return render_template('upload.html')

@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file'] #获取数据流
        rootPath = os.path.dirname(os.path.abspath(__file__)) #根目录路径
        #创建存储文件的文件夹，使用时间戳防止重名覆盖
        file_path = 'static/upload/' + str(int(time.time()))
        absolute_path = os.path.join(rootPath,file_path).replace('\\','/') #存储文件的绝对路径，window路径显示\\要转化/
        if not os.path.exists(absolute_path): #不存在改目录则会自动创建
            os.makedirs(absolute_path)
        save_file_name = os.path.join(absolute_path,f.filename).replace('\\','/') #文件存储路径（包含文件名）
        f.save(save_file_name)
        score=request.values.to_dict().get("score")
        num,fps=infer.inference(save_file_name,score)

        #return json.dumps({'code':200,'url':url_path},ensure_ascii=False)
        return render_template("show.html",num=num,fps=fps)

app.run(port='5000',debug=True)
```

## YOLO集成推理

而在YOLO集成的RT-DETR项目中，训练得到的权重 文件为.pt，在推理时需要与RT-DETR搭配使用，从而实现推理过程：
需要注意的是，由于YOLO里面集成了多种模型，因此为了具有适配性，其代码都具有通用性
```python
from ultralytics.models import RTDETR
if __name__ == '__main__':
    model=RTDETR("weights/best.pt")
    model.predict(source="images/1.mp4",save=True,conf=0.6)
```
随后执行`predict`，代码如下：

```python
def predict(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs,
    ) -> list:
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = predictor or self._smart_load("predictor")(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)
```
这部分代码在功能上具有复用性，因此在理解上存在一定难度。
