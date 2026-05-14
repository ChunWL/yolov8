from ultralytics import YOLO

# 🔥 密集人群 CrowdHuman 训练
if __name__ == '__main__':
    # 加载你之前训练好的模型
    model = YOLO("runs/detect/train/weights/best.pt")

    # 开始训练
    model.train(
        data="D:/python study/pythonProject/opencv/crowdhuman.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        lr0=0.0005,
        augment=True,
        patience=5,
        name="FEOD-YOLOv11-CrowdHuman"
    )