import cv2
import os
import glob
# import numpy as np
import random
import shutil
from pathlib import Path
import yaml


# ===============================================
# 第一部分：图像标注工具
# ===============================================

class MeterAnnotationTool:
    def __init__(self, image_folder):
        """
        电表读数区域标注工具

        Args:
            image_folder: 包含灰度电表图片的文件夹路径
        """
        self.image_folder = image_folder
        self.annotations_folder = "annotations"
        self.current_image = None
        self.current_image_path = ""
        self.image_list = []
        self.current_index = 0
        self.bbox_list = []
        self.drawing = False
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.temp_image = None

        # 创建标注文件夹
        os.makedirs(self.annotations_folder, exist_ok=True)

        # 加载图片列表
        self.load_image_list()

        # 类别信息
        self.class_names = ["meter_display"]
        self.current_class = 0

    def load_image_list(self):
        """加载图片文件列表"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        for ext in extensions:
            self.image_list.extend(glob.glob(os.path.join(self.image_folder, ext)))
            self.image_list.extend(glob.glob(os.path.join(self.image_folder, ext.upper())))

        if not self.image_list:
            print(f"错误：在文件夹 {self.image_folder} 中没有找到图片文件!")
            return

        self.image_list.sort()
        print(f"找到 {len(self.image_list)} 张图片待标注")

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_image = self.current_image.copy()
                cv2.rectangle(self.temp_image, self.start_point, (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)

            # 确保坐标顺序正确
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])

            # 检查边界框是否有效
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                self.bbox_list.append([x1, y1, x2, y2, self.current_class])
                print(f"添加边界框: ({x1}, {y1}, {x2}, {y2})")

            self.draw_bboxes()

    def draw_bboxes(self):
        """绘制所有边界框"""
        self.current_image = cv2.imread(self.current_image_path)

        for bbox in self.bbox_list:
            x1, y1, x2, y2, class_id = bbox
            cv2.rectangle(self.current_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.current_image, self.class_names[class_id],
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def convert_to_yolo_format(self, bbox, img_width, img_height):
        """转换为YOLO格式"""
        x1, y1, x2, y2, class_id = bbox

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        # 归一化
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height

        return class_id, center_x, center_y, width, height

    def save_annotations(self):
        """保存标注"""
        if not self.bbox_list:
            print("当前图片没有标注")
            return

        img = cv2.imread(self.current_image_path)
        img_height, img_width = img.shape[:2]

        image_name = Path(self.current_image_path).stem
        label_file = os.path.join(self.annotations_folder, f"{image_name}.txt")

        with open(label_file, 'w') as f:
            for bbox in self.bbox_list:
                class_id, center_x, center_y, width, height = self.convert_to_yolo_format(
                    bbox, img_width, img_height)
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

        print(f"保存: {label_file}")

    def load_existing_annotations(self):
        """加载已有标注"""
        image_name = Path(self.current_image_path).stem
        label_file = os.path.join(self.annotations_folder, f"{image_name}.txt")

        self.bbox_list = []

        if os.path.exists(label_file):
            img = cv2.imread(self.current_image_path)
            img_height, img_width = img.shape[:2]

            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1]) * img_width
                        center_y = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height

                        x1 = int(center_x - width / 2)
                        y1 = int(center_y - height / 2)
                        x2 = int(center_x + width / 2)
                        y2 = int(center_y + height / 2)

                        self.bbox_list.append([x1, y1, x2, y2, class_id])

    def show_help(self):
        """显示帮助信息"""
        help_text = """
        ========== 电表读数区域标注工具 ==========

        操作说明:
        🖱️  鼠标左键拖拽: 绘制边界框（框选数字显示区域）
        💾 's' 键: 保存当前图片的标注
        ➡️  'n' 键: 下一张图片
        ⬅️  'p' 键: 上一张图片
        🗑️  'c' 键: 清除当前图片的所有标注
        ❌ 'd' 键: 删除最后一个标注
        ❓ 'h' 键: 显示帮助信息
        🚪 'q' 键: 退出程序

        标注提示:
        📍 请准确框选电表数字显示区域
        📏 边界框应该紧贴数字边缘
        🔢 可以标注多个数字区域

        ==========================================
        """
        print(help_text)

    def run(self):
        """运行标注工具"""
        if not self.image_list:
            return

        self.show_help()

        cv2.namedWindow('电表标注工具', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('电表标注工具', 1200, 800)
        cv2.setMouseCallback('电表标注工具', self.mouse_callback)

        # 加载第一张图片
        self.current_image_path = self.image_list[self.current_index]
        self.load_existing_annotations()
        self.draw_bboxes()

        while True:
            # 显示信息
            info = f"图片 {self.current_index + 1}/{len(self.image_list)} | " \
                   f"标注: {len(self.bbox_list)} 个 | " \
                   f"文件: {os.path.basename(self.current_image_path)}"

            display_img = self.current_image.copy() if self.temp_image is None else self.temp_image.copy()
            cv2.putText(display_img, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('电表标注工具', display_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_annotations()
            elif key == ord('n'):
                if self.current_index < len(self.image_list) - 1:
                    self.current_index += 1
                    self.current_image_path = self.image_list[self.current_index]
                    self.load_existing_annotations()
                    self.draw_bboxes()
                    self.temp_image = None
            elif key == ord('p'):
                if self.current_index > 0:
                    self.current_index -= 1
                    self.current_image_path = self.image_list[self.current_index]
                    self.load_existing_annotations()
                    self.draw_bboxes()
                    self.temp_image = None
            elif key == ord('c'):
                self.bbox_list = []
                self.draw_bboxes()
                print("清除所有标注")
            elif key == ord('d'):
                if self.bbox_list:
                    self.bbox_list.pop()
                    self.draw_bboxes()
                    print("删除最后一个标注")
            elif key == ord('h'):
                self.show_help()

        cv2.destroyAllWindows()


# ===============================================
# 第二部分：数据集准备
# ===============================================

def prepare_dataset(images_folder, annotations_folder, output_folder="dataset", train_ratio=0.8):
    """准备YOLO训练数据集"""

    print("📁 准备YOLO数据集...")

    # 创建目录结构
    dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    for dir_path in dirs:
        os.makedirs(os.path.join(output_folder, dir_path), exist_ok=True)

    # 获取图片和标注文件对
    image_files = []
    label_files = []

    for img_path in Path(images_folder).glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            label_path = Path(annotations_folder) / f"{img_path.stem}.txt"
            if label_path.exists():
                image_files.append(img_path)
                label_files.append(label_path)

    if not image_files:
        print("❌ 没有找到匹配的图片-标注对！")
        return False

    print(f"✅ 找到 {len(image_files)} 对图片-标注文件")

    # 随机打乱并分割
    combined = list(zip(image_files, label_files))
    random.shuffle(combined)

    train_count = int(len(combined) * train_ratio)
    train_data = combined[:train_count]
    val_data = combined[train_count:]

    print(f"📊 训练集: {len(train_data)} 张")
    print(f"📊 验证集: {len(val_data)} 张")

    # 复制文件
    def copy_data(data_list, split):
        for img_path, label_path in data_list:
            # 复制图片
            dst_img = os.path.join(output_folder, 'images', split, img_path.name)
            shutil.copy2(img_path, dst_img)

            # 复制标注
            dst_label = os.path.join(output_folder, 'labels', split, label_path.name)
            shutil.copy2(label_path, dst_label)

    copy_data(train_data, 'train')
    copy_data(val_data, 'val')

    # 创建配置文件
    config = {
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['meter_display']
    }

    config_path = os.path.join(output_folder, 'dataset.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ 数据集准备完成: {output_folder}")
    print(f"📄 配置文件: {config_path}")

    return True


# ===============================================
# 第三部分：模型训练
# ===============================================

def install_requirements():
    """安装必要的依赖"""
    print("📦 检查并安装依赖包...")

    requirements = [
        "ultralytics",
        "opencv-python",
        "PyYAML",
        "torch",
        "torchvision"
    ]

    install_commands = []
    for package in requirements:
        install_commands.append(f"pip install {package}")

    print("请运行以下命令安装依赖:")
    for cmd in install_commands:
        print(f"  {cmd}")

    print("\n或者一次性安装:")
    print(f"  pip install {' '.join(requirements)}")


def train_yolo_model(dataset_folder="dataset", epochs=100, img_size=640, batch_size=16):
    """训练YOLO模型"""

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ 未安装ultralytics，请先安装依赖")
        install_requirements()
        return False

    print("🚀 开始训练YOLO模型...")

    # 检查数据集配置文件
    config_path = os.path.join(dataset_folder, 'dataset.yaml')
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False

    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 使用YOLOv8 nano模型

    # 训练参数
    train_args = {
        'data': config_path,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'name': 'meter_detection',
        'save': True,
        'save_period': 10,
        'val': True,
        'plots': True,
        'device': 'auto'  # 自动选择GPU或CPU
    }

    print("🔧 训练参数:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")

    try:
        # 开始训练
        results = model.train(**train_args)

        print("✅ 训练完成!")
        print(f"📁 模型保存在: runs/detect/meter_detection/weights/")
        print(f"📊 最佳模型: runs/detect/meter_detection/weights/best.pt")

        return True

    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        return False


# ===============================================
# 第四部分：模型测试
# ===============================================

def test_model(model_path="runs/detect/meter_detection/weights/best.pt", test_images_folder="processed_images"):
    """测试训练好的模型"""

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ 未安装ultralytics")
        return

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return

    print(f"🔍 加载模型: {model_path}")
    model = YOLO(model_path)

    # 获取测试图片
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(glob.glob(os.path.join(test_images_folder, ext)))

    if not test_images:
        print(f"❌ 在 {test_images_folder} 中没有找到测试图片")
        return

    print(f"🖼️ 找到 {len(test_images)} 张测试图片")

    # 创建结果文件夹
    results_folder = "detection_results"
    os.makedirs(results_folder, exist_ok=True)

    # 输出表头
    print("\n检测结果:")
    print("filename\t\txmin\tymin\txmax\tymax")
    print("-" * 50)

    # 预测并保存结果
    for img_path in test_images[:50]:  # 只测试前50张
        print(f"处理: {os.path.basename(img_path)}")

        results = model(img_path)

        # 保存结果图片
        for i, result in enumerate(results):
            result_img = result.plot()
            result_path = os.path.join(results_folder, f"result_{os.path.basename(img_path)}")
            cv2.imwrite(result_path, result_img)

            # 输出检测框位置信息
            filename = os.path.basename(img_path)

            if result.boxes is not None and len(result.boxes) > 0:
                # 获取检测框坐标
                boxes = result.boxes.xyxy.cpu().numpy()  # 转换为numpy数组

                for box in boxes:
                    xmin, ymin, xmax, ymax = box[:4]
                    print(f"{filename}\t\t{int(xmin)}\t{int(ymin)}\t{int(xmax)}\t{int(ymax)}")
            else:
                print(f"{filename}\t\t未检测到目标")

    print(f"\n✅ 检测结果保存在: {results_folder}")


# ===============================================
# 主程序
# ===============================================

def main():
    """主程序入口"""

    print("=" * 60)
    print("🔬 电表读数区域检测完整流程")
    print("=" * 60)

    # 检查图片文件夹
    images_folder = "processed_images"
    if not os.path.exists(images_folder):
        print(f"❌ 图片文件夹 '{images_folder}' 不存在!")
        print("请确保将灰度电表图片放在该文件夹中")
        return

    while True:
        print("\n🔧 选择操作:")
        print("1. 📝 标注图片 (创建训练标签)")
        print("2. 📁 准备数据集 (整理为YOLO格式)")
        print("3. 🚀 训练模型")
        print("4. 🔍 测试模型")
        print("5. 📦 安装依赖")
        print("0. 🚪 退出")

        choice = input("\n请选择 (0-5): ").strip()

        if choice == '1':
            print("\n📝 启动图片标注工具...")
            annotator = MeterAnnotationTool(images_folder)
            annotator.run()

        elif choice == '2':
            print("\n📁 准备数据集...")
            success = prepare_dataset(images_folder, "annotations")
            if success:
                print("✅ 数据集准备完成!")

        elif choice == '3':
            print("\n🚀 开始训练模型...")
            epochs = input("训练轮数 (默认100): ").strip()
            epochs = int(epochs) if epochs.isdigit() else 100

            success = train_yolo_model(epochs=epochs)
            if success:
                print("✅ 模型训练完成!")

        elif choice == '4':
            print("\n🔍 测试模型...")
            model_path = input("模型路径 (默认: runs/detect/meter_detection/weights/best.pt): ").strip()
            if not model_path:
                model_path = "runs/detect/meter_detection/weights/best.pt"
            test_model(model_path)

        elif choice == '5':
            install_requirements()

        elif choice == '0':
            print("👋 再见!")
            break

        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()