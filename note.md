
download docker image:

```bash
sudo docker pull nvcr.io/nvidia/pytorch:21.03-py3
```

run:
```bash
sudo docker run --gpus all -it  -v $pwd:/code nvcr.io/nvidia/pytorch:21.03-py3


sudo nvidia-docker run -it --name yolov5-test --gpus all -v $PWD:/workplace -w /workplace nvcr.io/nvidia/pytorch:21.03-py3 bash




sudo nvidia-docker run -it --name yolov5-test --gpus all -v $PWD:/workplace -w /workplace nvcr.io/nvidia/pytorch:21.03-py3 bash

sudo nvidia-docker run -it --name yolov5-test --gpus all -v $PWD:/workplace -w /workplace pytorch-py38-cuda11.2-cudnn8-ubuntu20.04 bash

sudo nvidia-docker run -it --name yolov5-test --gpus all -v $PWD:/workplace -w /workplace nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu18.04 bash


docker pull pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime


https://blog.csdn.net/qq_27036771/article/details/107316314

docker pull nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04
```

---
[修改版本的仓库地址](https://github.com/EASY-EAI/yolov5)
> 注：yolov5工程需要使用pytorch 1.8.0 或 1.9.0 版本才能正常导出



yolov5代码环境依赖:
```
  torch==1.9.0  
  torchvision==0.10.0
```

模型转换环境依赖：
```
rk-toolkit使用的是1.7.1
Python=3.6, 
onnx==1.6.0             # rk-toolkit限制onnx必须是1.6.0
torch==1.9.0        
torchvision==0.10.0
```


输入图片尺寸 416x416

训练出的模型可以识别

### 导出模型
>导出模型可选择去掉尾端的permute层，从而兼容rknn_yolov5_demo的c++部署代码
```bash
python3 models/export.py --rknn_mode --ignore_output_permute
```

我之前第一次转换时，发现是我的输出格式不是`1*24*20*20`，而是成了例如`1*3*20*20*8`这种，于是我在用export.py生成onnx模型前，把yolov5-5.0工程中的yolo.py做了如下修改(仅仅是用export时这样改，train以及detect时候不要用这个yolo.py)，然后再用export.py生成onnx模型，然后再转成rknn模型即可。

工程中的yolo.py做了如下修改(仅仅是用export时这样改，train以及detect时候不要用这个yolo.py)，然后再用export.py生成onnx模型，然后再转成rknn模型即可。
`models/yolo.py`
首先把export改成True
```python
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    export = True   #改成True
```

然后forward代码这里做如下修改
```python
def forward(self, x):
    # x = x.copy()  # for profiling
    z = []  # inference output
    self.training |= self.export
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv

        if self.ignore_permute_layer is True:
            continue

        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        #x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        x[i] = x[i].view(bs, self.na*self.no, ny, nx).contiguous()

        if not self.training:  # inference
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
```
注意修改是把permute去掉，并且view()里面是把self.na和self.no乘起来了。

或者按照如下方式修改，把self.training |= self.export注释掉，然后直接添加一个export分支：
```python
  def forward(self, x):
      # x = x.copy()  # for profiling
      z = []  # inference output
      # self.training |= self.export
      if self.export:
          for i in range(self.nl):
              x[i] = self.m[i](x[i])
              bs, _, ny, nx = x[i].shape  # x(bs,48,20,20) to x(bs,3,20,20,16)
              x[i] = x[i].view(bs, self.na*self.no, ny, nx).contiguous()

          return x

      for i in range(self.nl):
          x[i] = self.m[i](x[i])  # conv
          bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
          x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
      .......后面代码........
```



### 模型转换
通过Rk的官方转换示例转换模型
```
https://github.com/rockchip-linux/rknpu/blob/master/rknn/rknn_api/examples/rknn_yolov5_demo/convert_rknn_demo/yolov5/onnx2rknn.py
```

### 编译在1808上运行的程序

配置好交叉编译的工具链
```
https://github.com/rockchip-linux/rknpu/blob/master/rknn/rknn_api/examples/rknn_yolov5_demo/build.sh
```


将install目录下的文件拷贝到1808上就可以实现预测