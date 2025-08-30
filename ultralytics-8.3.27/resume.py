from ultralytics import YOLO

#断点续训启动方式,这个绝对路径就是上文中提到的last.pt文件路径
model = YOLO("/root/ultralytics-8.3.27/runs/detect/train2/weights/last.pt")
# 中断训练的权重文件中的last.pt
results = model.train(resume=True)

