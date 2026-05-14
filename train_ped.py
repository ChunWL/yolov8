from ultralytics import YOLO

# Windows 显卡训练必须加这一行！
if __name__ == '__main__':

    # 加载预训练模型（效果直接起飞）
    model = YOLO("yolov8n.pt")

    model.train(
        data=r"D:\python study\pythonProject\opencv\mydata.yaml",
        epochs=100,
        batch=8,
        imgsz=640,
        device=0,     
        workers=0,    
        plots=False
    )