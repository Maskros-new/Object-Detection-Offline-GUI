# Object-Detection-Offline-GUI 
基于候选区域的目标检测器-- 利用pyqt实现GUI

version 0.2.0

## REQUIREMENTS

> tips: Keep `torch`, `torchvision`, `pillow`the same as the version when training the model

- `torch1.8.0+cu111`
- `torchvision0.9.0+cu111`
- `pillow7.0.0`
- `PyQt5`
- `pyinstaller` (optional)

## PROCESS

> 模型训练过程：[link](https://github.com/willchao612/ObjDetApp)

- 需要用到训练好的模型`.pth`文件

- 由于最终目标是面对所有windows用户，故设备设置为 cpu

  ```python
  device = torch.device("cpu")
  ```

- 为防止导入图片过大，对图片从长宽进行判断做`resize()`处理

  ```python
  original_image = original_image.resize((width, height), Image.ANTIALIAS)
  ```

- 使用`pyinstaller`进行打包，链接`.pth` 静态文件

  ```
  pyinstaller -F --add-data ".\model_state_dict.pth;," ".\v0.2.py"
  ```

## DEMO

![img](https://s3.bmp.ovh/imgs/2021/12/4455efdffb94d08a.png)

