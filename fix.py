import os
#处理crowdhuman标签不止行人的问题（我们要的是行人检测）
#  你的路径
base_path = r"D:\python study\pythonProject\opencv\CrowdHuman"

def clean_labels(folder_path):
    total_files = 0
    fixed_files = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as f:
                lines = f.readlines()

            new_lines = []
            changed = False
            for line in lines:
                # 只保留 类别 0（person），删掉 1、2、其他所有类别
                if line.startswith("0 "):
                    new_lines.append(line)
                else:
                    changed = True

            if changed:
                with open(filepath, "w") as f:
                    f.writelines(new_lines)
                fixed_files += 1
            total_files += 1

    print(f"✅ 清理完成 {folder_path}")
    print(f"📊 总文件：{total_files} | 修复文件：{fixed_files}")

# 清理训练集 + 验证集标签
clean_labels(os.path.join(base_path, "labels/train"))
clean_labels(os.path.join(base_path, "labels/val"))

print("\n🎉 所有标签已清理完成！只保留 人(person=0) 类别！")