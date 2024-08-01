## 铁路病害检测软件
本项目使用pyqt编写，对pyqt5版本有要求。
#### 使用注意事项
1. tools中为脚本文件，开启ui后自动开启，关闭ui自动关闭(非bug引起的关闭)，若非正常关闭，可以通过 `nvidia-smi`观察PID直接kill掉。
2. 脚本分为动态实时监测脚本与静态预标注脚本，使用预表注脚本需要通过打开文件目录后，点击右侧批量标注或单张标注进行执行，主程序会把打开的文件路径给脚本，脚本自动生成json到该文件路径下。
3. 实时监测脚本，通过打开实时监测目录，在目录中的文件夹添加图片时，程序自动将图片呈现，若打开自动检测该图片便会输入脚本中开始检测。
#### 环境
pyqt5 mmdetection 3.0.0以上 
#### bug
1. mmdet与pyqt无法共存，使用脚本方式在不同进程运行。
2. 使用gsw的服务器。运行脚本之前需要输入 `unset LD_LIBRARY_PATH`
3. 对pyqt版本有要求，在部分电脑上（另一台服务器）会报bug（浮点数问题），适当降版本使用。
4. linux版本为5.15.9 ``if event.button() == Qt.LeftButton:``
            ``item = self.get_item_at_click(event)``
            ``# item.setSelected(True) #报错？？？``
    
