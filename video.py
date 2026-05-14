from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO(r"runs\detect\FEOD-YOLOv11-CrowdHuman5\weights\best.pt")#用的哪个模型,改成你的路径

# ================= 抗闪烁关键参数（可微调）================
IOU_THRESH    = 0.3
SMOOTH_ALPHA  = 0.65   # 越大越稳
VANISH_DELAY  =  0      # 框消失后再多保留8帧 = 彻底不闪
CONF_MIN      = 0.35   # 提高置信，干掉闪的碎框
# =========================================================

def calc_iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    inter_x1 = max(x1, a1)
    inter_y1 = max(y1, b1)
    inter_x2 = min(x2, a2)
    inter_y2 = min(y2, b2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area

tracks = []

if __name__ == '__main__':
    cap = cv2.VideoCapture("D:/python study/pythonProject/opencv/shipin/people4.avi")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    writer = cv2.VideoWriter("final_no_flash.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推理：高置信去闪
        res = model(
            frame,
            conf=CONF_MIN,
            iou=0.45,
            imgsz=960,
            verbose=False
        )[0]

        curr_boxes = []
        if res.boxes is not None:
            curr_boxes = res.boxes.xyxy.cpu().numpy().tolist()

        new_tracks = []
        used_idx = [False] * len(curr_boxes)

        # 1. 匹配旧轨迹 + 平滑 + 续命
        for tr in tracks:
            best_iou = 0
            best_pos = -1
            for i, cb in enumerate(curr_boxes):
                if used_idx[i]:
                    continue
                iou_val = calc_iou(tr["box"], cb)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_pos = i

            if best_iou > IOU_THRESH and best_pos != -1:
                # 同目标平滑融合
                old = np.array(tr["box"])
                now = np.array(curr_boxes[best_pos])
                smooth_box = old * SMOOTH_ALPHA + now * (1 - SMOOTH_ALPHA)
                tr["box"] = smooth_box.tolist()
                tr["life"] = VANISH_DELAY   # 刷新续命
                new_tracks.append(tr)
                used_idx[best_pos] = True
            else:
                # 目标没匹配到 → 倒计时消失（延时消影）
                tr["life"] -= 1
                if tr["life"] > 0:
                    new_tracks.append(tr)

        # 2. 新增当前新检测目标
        for i, cb in enumerate(curr_boxes):
            if not used_idx[i]:
                new_tracks.append({"box": cb, "life": VANISH_DELAY})

        tracks = new_tracks

        # 绘制所有续命中的框（杜绝一闪没）
        for t in tracks:
            x1, y1, x2, y2 = map(int, t["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        writer.write(frame)
        cv2.imshow("加延时消影｜无闪烁", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()