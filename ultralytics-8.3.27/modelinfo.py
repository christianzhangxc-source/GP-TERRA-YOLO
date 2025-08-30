from ultralytics import YOLO

# 加载模型
model = YOLO('/root/ultralytics-8.3.27/runs/detect/train3/weights/best.pt')

# 打印模型信息
model.info()