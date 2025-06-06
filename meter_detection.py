import cv2
import os
import glob
# import numpy as np
import random
import shutil
from pathlib import Path
import yaml
import csv
from datetime import datetime

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
# 第四部分：模型测试（只取最高置信度结果）
# ===============================================

def test_model(model_path="runs/detect/meter_detection/weights/best.pt", test_images_folder="processed_images"):
    """测试训练好的模型 - 只保留置信度最高的检测结果"""

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
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    for ext in extensions:
        test_images.extend(glob.glob(os.path.join(test_images_folder, ext)))
        test_images.extend(glob.glob(os.path.join(test_images_folder, ext.upper())))

    if not test_images:
        print(f"❌ 在 {test_images_folder} 中没有找到测试图片")
        return

    print(f"🖼️ 找到 {len(test_images)} 张测试图片")

    # 创建结果文件夹
    results_folder = "detection_results"
    os.makedirs(results_folder, exist_ok=True)

    # 准备CSV文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(results_folder, f"detection_results_best_{timestamp}.csv")

    # CSV数据存储列表
    csv_data = []

    # 添加CSV表头
    csv_headers = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class_name', 'total_detections']

    # 输出表头到控制台
    print("\n检测结果 (仅显示最高置信度):")
    print("filename\t\txmin\tymin\txmax\tymax\tconfidence\ttotal_det")
    print("-" * 80)

    # 统计变量
    total_images_with_detection = 0
    total_images_with_multiple_detection = 0
    processed_count = 0

    # 预测并保存结果
    for img_path in test_images:  # 测试所有图片
        filename = os.path.basename(img_path)
        print(f"处理 ({processed_count + 1}/{len(test_images)}): {filename}")

        try:
            results = model(img_path)

            # 处理检测结果
            best_detection = None
            total_detections_count = 0

            for i, result in enumerate(results):
                if result.boxes is not None and len(result.boxes) > 0:
                    # 获取检测框坐标、置信度和类别
                    boxes = result.boxes.xyxy.cpu().numpy()  # 坐标
                    confidences = result.boxes.conf.cpu().numpy()  # 置信度
                    classes = result.boxes.cls.cpu().numpy()  # 类别

                    total_detections_count = len(boxes)

                    if total_detections_count > 0:
                        total_images_with_detection += 1

                        if total_detections_count > 1:
                            total_images_with_multiple_detection += 1

                        # 找到置信度最高的检测结果
                        best_idx = confidences.argmax()  # 获取置信度最高的索引

                        best_box = boxes[best_idx]
                        best_confidence = confidences[best_idx]
                        best_class_id = int(classes[best_idx])
                        best_class_name = model.names[best_class_id] if best_class_id in model.names else 'unknown'

                        xmin, ymin, xmax, ymax = best_box[:4]

                        best_detection = {
                            'filename': filename,
                            'xmin': int(xmin),
                            'ymin': int(ymin),
                            'xmax': int(xmax),
                            'ymax': int(ymax),
                            'confidence': best_confidence,
                            'class_name': best_class_name,
                            'total_detections': total_detections_count
                        }

                # 保存可视化结果图片（显示最佳检测）
                if best_detection:
                    result_img = result.plot()  # 这会显示所有检测结果

                    # 创建只显示最佳检测的图片
                    original_img = cv2.imread(img_path)
                    cv2.rectangle(original_img,
                                  (best_detection['xmin'], best_detection['ymin']),
                                  (best_detection['xmax'], best_detection['ymax']),
                                  (0, 255, 0), 3)  # 绿色粗线框

                    # 添加标签
                    label = f"BEST: {best_detection['class_name']} {best_detection['confidence']:.3f}"
                    cv2.putText(original_img, label,
                                (best_detection['xmin'], best_detection['ymin'] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 添加总检测数信息
                    info_text = f"Total detections: {total_detections_count}"
                    cv2.putText(original_img, info_text,
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # 保存最佳检测结果图片
                    result_path = os.path.join(results_folder, f"best_{filename}")
                    cv2.imwrite(result_path, original_img)

            # 添加到CSV数据和控制台输出
            if best_detection:
                csv_data.append([
                    best_detection['filename'],
                    best_detection['xmin'],
                    best_detection['ymin'],
                    best_detection['xmax'],
                    best_detection['ymax'],
                    f"{best_detection['confidence']:.4f}",
                    best_detection['class_name'],
                    best_detection['total_detections']
                ])

                # 输出到控制台
                print(f"{filename}\t\t{best_detection['xmin']}\t{best_detection['ymin']}\t"
                      f"{best_detection['xmax']}\t{best_detection['ymax']}\t"
                      f"{best_detection['confidence']:.4f}\t\t{total_detections_count}")

                # 如果有多个检测，显示提示
                if total_detections_count > 1:
                    print(f"  └─ 注意: 共检测到 {total_detections_count} 个目标，已选择置信度最高的")
            else:
                # 没有检测到目标的情况
                csv_data.append([
                    filename,
                    '',
                    '',
                    '',
                    '',
                    '',
                    'no_detection',
                    0
                ])
                print(f"{filename}\t\t未检测到目标")

        except Exception as e:
            print(f"❌ 处理图片 {filename} 时出错: {e}")
            # 记录错误到CSV
            csv_data.append([
                filename,
                '',
                '',
                '',
                '',
                '',
                'error',
                0
            ])

        processed_count += 1

    # 写入CSV文件
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)  # 写入表头
            writer.writerows(csv_data)  # 写入数据

        print(f"\n✅ CSV结果已保存: {csv_filename}")
    except Exception as e:
        print(f"❌ 保存CSV文件时出错: {e}")

    # 统计信息
    print(f"\n📊 检测统计:")
    print(f"  总图片数: {len(test_images)}")
    print(f"  处理成功: {processed_count}")
    print(f"  有检测结果: {total_images_with_detection}")
    print(
        f"  检测成功率: {(total_images_with_detection / processed_count) * 100:.2f}%" if processed_count > 0 else "  检测成功率: 0%")
    print(f"  多目标检测图片: {total_images_with_multiple_detection}")
    print(
        f"  多目标率: {(total_images_with_multiple_detection / max(total_images_with_detection, 1)) * 100:.2f}%" if total_images_with_detection > 0 else "  多目标率: 0%")
    print(f"📁 最佳检测结果图片保存在: {results_folder}/best_*.jpg")
    print(f"📄 详细结果保存在: {csv_filename}")


def analyze_detection_results(csv_file_path):
    """分析检测结果的统计信息"""

    if not os.path.exists(csv_file_path):
        print(f"❌ CSV文件不存在: {csv_file_path}")
        return

    print(f"📊 分析检测结果: {csv_file_path}")

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        # 基本统计
        total_images = len(rows)
        successful_detections = [row for row in rows if row['class_name'] not in ['no_detection', 'error', '']]
        no_detection_count = len([row for row in rows if row['class_name'] == 'no_detection'])
        error_count = len([row for row in rows if row['class_name'] == 'error'])

        # 多检测统计
        multi_detection_images = [row for row in rows if row['total_detections'] and int(row['total_detections']) > 1]

        # 置信度统计
        confidences = []
        for row in successful_detections:
            if row['confidence']:
                try:
                    confidences.append(float(row['confidence']))
                except ValueError:
                    pass

        print(f"\n📈 检测结果统计:")
        print(f"  总图片数: {total_images}")
        print(f"  成功检测数: {len(successful_detections)}")
        print(f"  未检测到: {no_detection_count}")
        print(f"  处理错误: {error_count}")
        print(
            f"  检测成功率: {(len(successful_detections) / total_images) * 100:.2f}%" if total_images > 0 else "  检测成功率: 0%")

        # 多检测分析
        print(f"\n🔍 多检测分析:")
        print(f"  有多个检测的图片: {len(multi_detection_images)}")
        print(
            f"  多检测率: {(len(multi_detection_images) / len(successful_detections)) * 100:.2f}%" if successful_detections else "  多检测率: 0%")

        if multi_detection_images:
            detection_counts = [int(row['total_detections']) for row in multi_detection_images]
            print(f"  最多检测数: {max(detection_counts)}")
            print(f"  平均检测数: {sum(detection_counts) / len(detection_counts):.2f}")

        # 置信度分析
        if confidences:
            print(f"\n🎯 置信度分析:")
            print(f"  平均置信度: {sum(confidences) / len(confidences):.4f}")
            print(f"  最高置信度: {max(confidences):.4f}")
            print(f"  最低置信度: {min(confidences):.4f}")

            # 置信度分布
            high_conf = len([c for c in confidences if c >= 0.8])
            medium_conf = len([c for c in confidences if 0.5 <= c < 0.8])
            low_conf = len([c for c in confidences if c < 0.5])

            print(f"\n📊 置信度分布:")
            print(f"  高置信度 (≥0.8): {high_conf} ({high_conf / len(confidences) * 100:.1f}%)")
            print(f"  中置信度 (0.5-0.8): {medium_conf} ({medium_conf / len(confidences) * 100:.1f}%)")
            print(f"  低置信度 (<0.5): {low_conf} ({low_conf / len(confidences) * 100:.1f}%)")

    except Exception as e:
        print(f"❌ 分析CSV文件时出错: {e}")


def show_multi_detection_samples(csv_file_path, top_n=10):
    """显示多检测样本"""

    if not os.path.exists(csv_file_path):
        print(f"❌ CSV文件不存在: {csv_file_path}")
        return

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        # 找出多检测的图片，按检测数量排序
        multi_detection_rows = [row for row in rows if row['total_detections'] and int(row['total_detections']) > 1]
        multi_detection_rows.sort(key=lambda x: int(x['total_detections']), reverse=True)

        print(f"\n🔍 多检测样本 (前{min(top_n, len(multi_detection_rows))}个):")
        print("-" * 100)

        for i, row in enumerate(multi_detection_rows[:top_n]):
            print(f"{i + 1:2d}. {row['filename']} - 检测数: {row['total_detections']}, "
                  f"最佳置信度: {row['confidence']}, "
                  f"坐标: ({row['xmin']},{row['ymin']})-({row['xmax']},{row['ymax']})")

        if multi_detection_rows:
            print(f"\n💡 建议查看这些图片的 best_*.jpg 结果文件，确认检测质量")

    except Exception as e:
        print(f"❌ 显示多检测样本时出错: {e}")

# 更新主程序中的选择项
def main():
    """主程序入口 - 更新版本"""

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
        print("4. 🔍 测试模型 (生成CSV结果)")
        print("5. 📊 分析检测结果")
        print("6. 📦 安装依赖")
        print("0. 🚪 退出")

        choice = input("\n请选择 (0-6): ").strip()

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
            print("\n📊 分析检测结果...")
            csv_files = glob.glob("detection_results/detection_results_*.csv")
            if csv_files:
                print("找到的CSV文件:")
                for i, file in enumerate(csv_files, 1):
                    print(f"  {i}. {os.path.basename(file)}")

                choice_csv = input(f"选择文件 (1-{len(csv_files)}) 或输入完整路径: ").strip()

                if choice_csv.isdigit() and 1 <= int(choice_csv) <= len(csv_files):
                    csv_file = csv_files[int(choice_csv) - 1]
                else:
                    csv_file = choice_csv

                analyze_detection_results(csv_file)
            else:
                print("❌ 没有找到检测结果CSV文件")
                csv_file = input("请输入CSV文件路径: ").strip()
                if csv_file:
                    analyze_detection_results(csv_file)

        elif choice == '6':
            install_requirements()

        elif choice == '0':
            print("👋 再见!")
            break

        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()