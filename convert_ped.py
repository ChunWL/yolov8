#处理数据集，转换标签
import os
import re

# 你的路径
IMG_DIR = r"D:\python study\pythonProject\opencv\datasets\PennFudanPed\PNGImages"
ANN_DIR = r"D:\python study\pythonProject\opencv\datasets\PennFudanPed\Annotation"

# 输出标准 YOLO 格式
OUT_BASE = r"D:\python study\pythonProject\opencv\mydata"
OUT_IMAGES = os.path.join(OUT_BASE, "images")
OUT_LABELS = os.path.join(OUT_BASE, "labels")
os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)

for txt_name in os.listdir(ANN_DIR):
    if not txt_name.endswith(".txt"):
        continue

    base = os.path.splitext(txt_name)[0]

    # 读取标注
    with open(os.path.join(ANN_DIR, txt_name), encoding="utf-8") as f:
        content = f.read()

    w = int(re.search(r"Image size.*?: (\d+) x", content).group(1))
    h = int(re.search(r"x (\d+) x", content).group(1))
    boxes = re.findall(r"\((\d+), (\d+)\) - \((\d+), (\d+)\)", content)

    # 写出 YOLO 标签
    with open(os.path.join(OUT_LABELS, base + ".txt"), "w") as f:
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            cx = (x1 + x2)/2/w
            cy = (y1 + y2)/2/h
            bw = (x2-x1)/w
            bh = (y2-y1)/h
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # 复制图片（保持同名）
    import shutil
    src_img = os.path.join(IMG_DIR, base + ".png")
    dst_img = os.path.join(OUT_IMAGES, base + ".png")
    shutil.copy(src_img, dst_img)

print("✅ 标准 YOLO 数据集构建完成！！！")