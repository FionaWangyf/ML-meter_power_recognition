# coding=utf-8
import cv2
import numpy as np
import os

script_dir = os.path.dirname(__file__)
input_folder = os.path.abspath(os.path.join(script_dir, "../../data/raw/Dataset"))
output_folder_2 = os.path.abspath(os.path.join(script_dir, "../../data/processed"))

os.makedirs(output_folder_2, exist_ok=True)
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):

        filepath = os.path.join(input_folder, filename)
        image = cv2.imread(filepath)
        if image is None:
            print(f"跳过无法读取的图像: {filename}")
            continue

        # === 亮斑增强 ===
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab_enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 鱼眼畸变矫正
        h, w = enhanced.shape[:2]
        K = np.array([[w, 0, w / 2],
                      [0, w, h / 2],
                      [0, 0, 1]], dtype=np.float32)
        D = np.array([-0.5, 0.2, 0, 0], dtype=np.float32)

        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
        undistorted = cv2.remap(enhanced, map1, map2, interpolation=cv2.INTER_LINEAR)

        # 保存
        # output_path1 = os.path.join(output_folder_1, filename)
        # cv2.imwrite(output_path1, undistorted)

        # 灰化处理
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        output_path2 = os.path.join(output_folder_2, filename)
        cv2.imwrite(output_path2, gray)

        print(f"处理完成: {filename}")

print("所有图像处理完成 ")
