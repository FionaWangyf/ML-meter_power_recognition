## 🗂️ 项目结构
```plaintext
digit_location/
├── 📁 annotations/              # 原始标注文件
│   └── *.txt                   # 训练集标注文件（自定义格式）
├── 📁 dataset/                 # YOLO格式数据集
│   ├── 📁 images/
│   │   ├── 📁 train/           # 训练图像
│   │   └── 📁 val/             # 验证图像
│   ├── 📁 labels/
│   │   ├── 📁 train/           # 训练标签（YOLO格式）
│   │   └── 📁 val/             # 验证标签（YOLO格式）
│   └── 📄 dataset.yaml         # 数据集配置文件
├── 📁 detection_results/       # 模型测试结果
│   └── *.jpg                   # 检测结果图像
├── 📁 processed_images/        # 预处理图像
│   └── *.jpg                   # 灰度化处理后的图像
├── 📁 runs/                    # 训练输出结果
│   └── detect/
│       └── meter_detection/    # 模型权重和训练日志
└── 📄 meter_detection.py       # 主程序脚本
```
训练好的最佳模型的路径：
runs/detect/meter_detection/weights/best.pt