from ultralytics import YOLO

if __name__ == '__main__':
    # 加载你最新训练的存档（续训必须用 last.pt）
    model = YOLO("runs/detect/FEOD-YOLOv11-CrowdHuman5/weights/last.pt")

    # 继续训练
    model.train(
        data="D:/python study/pythonProject/opencv/crowdhuman.yaml",
        epochs=50,        # 总轮数不变，还是50轮
        imgsz=640,
        batch=8,
        device=0,
        lr0=0.0005,
        cache=True,
        resume=True,      # 👈 这个就是【继续训练】
        name="FEOD-YOLOv11-CrowdHuman5"  # 👈 用同一个文件夹，不新建！
    )