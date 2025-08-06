#!/usr/bin/env python3
"""
牛只检测器示例脚本
演示如何使用CattleDetector类检测图像中的牛只并可视化结果
"""

import os
import sys
import argparse
import cv2
import json
import numpy as np
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# 导入牛只检测器
try:
    from src.image_recognition_server.cattle.detector import CattleDetector, DEPS_INSTALLED
    if not DEPS_INSTALLED:
        print("错误: 缺少依赖项。请安装必要的依赖:")
        print("pip install supervision ultralytics opencv-python")
        sys.exit(1)
except ImportError as e:
    print(f"错误: 导入模块失败 - {e}")
    print("请确保您已安装所有必要的依赖:")
    print("pip install supervision ultralytics opencv-python")
    sys.exit(1)

def visualize_detection(image, results, output_path):
    """在图像上可视化检测结果"""
    # 复制图像以避免修改原图
    vis_image = image.copy()
    
    # 在图像上绘制检测结果
    for i, result in enumerate(results):
        # 获取边界框坐标
        x1, y1, x2, y2 = [int(coord) for coord in result["box"]]
        confidence = result["confidence"]
        class_name = result["class_name"]
        
        # 绘制边界框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"#{i+1} {class_name}: {confidence:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # 绘制标签背景
        cv2.rectangle(
            vis_image, 
            (x1, y1 - 25), 
            (x1 + label_width, y1), 
            (0, 255, 0), 
            -1
        )
        
        # 绘制标签文本
        cv2.putText(
            vis_image, 
            label, 
            (x1, y1 - 7), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 0), 
            1
        )
    
    # 添加检测统计信息
    info_text = f"检测到 {len(results)} 头牛"
    cv2.putText(
        vis_image,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # 保存结果图像
    cv2.imwrite(output_path, vis_image)
    print(f"已保存可视化结果到: {output_path}")
    
    return vis_image

def load_image(image_path):
    """加载图像，支持本地路径和URL"""
    if image_path.startswith(('http://', 'https://')):
        # 从URL加载图像
        import requests
        try:
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"无法解码图像: {image_path}")
            return image
        except Exception as e:
            raise ValueError(f"无法从URL加载图像: {e}")
    else:
        # 从本地路径加载图像
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像文件: {image_path}")
        
        return image

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='牛只检测器示例')
    parser.add_argument('--image', '-i', required=False,
                        default=os.path.join(os.path.dirname(__file__), 'imgs', 'test1.png'),
                        help='输入图像路径或URL')
    parser.add_argument('--confidence', '-c', type=float, default=0.25,
                        help='检测置信度阈值 (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='NMS的IoU阈值 (0.0-1.0)')
    parser.add_argument('--output', '-o', 
                        default=os.path.join(os.path.dirname(__file__), 'cattle_detected.jpg'),
                        help='输出图像路径')
    parser.add_argument('--json', '-j', action='store_true',
                        help='以JSON格式输出详细结果')
    
    args = parser.parse_args()
    
    print("\n===== 牛只检测器示例 =====")
    print(f"输入图像: {args.image}")
    print(f"置信度阈值: {args.confidence}")
    print(f"IoU阈值: {args.iou}")
    print(f"输出图像: {args.output}")
    
    try:
        # 加载图像
        print("\n正在加载图像...")
        image = load_image(args.image)
        print(f"图像加载成功: {image.shape[1]}x{image.shape[0]} 像素")
        
        # 初始化检测器
        print("\n正在初始化检测器...")
        detector = CattleDetector(confidence=args.confidence, iou_threshold=args.iou)
        print("检测器初始化成功")
        
        # 执行检测
        print("\n正在执行检测...")
        start_time = time.time()
        results = detector.detect(image)
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"检测完成，用时 {elapsed:.4f} 秒")
        print(f"检测到 {len(results)} 头牛")
        
        # 显示检测结果
        if args.json:
            print("\n检测结果 (JSON格式):")
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print("\n检测结果摘要:")
            for i, result in enumerate(results):
                print(f"牛 #{i+1}: 置信度={result['confidence']:.2f}, 位置={[int(x) for x in result['box']]}")
        
        # 可视化检测结果
        visualize_detection(image, results, args.output)
        
        print("\n===== 示例运行完成 =====")
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)