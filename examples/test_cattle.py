#!/usr/bin/env python3
"""
牛只检测测试脚本
使用test1.png图片测试牛只检测功能
"""

import os
import sys
import cv2
import json
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def main():
    """主函数"""
    # 获取图像路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "imgs", "test1.png")
    output_path = os.path.join(script_dir, "test1_detected.jpg")
    
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return False
    
    print(f"测试图像: {image_path}")
    
    try:
        # 加载图像
        print("正在加载图像...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像文件: {image_path}")
            return False
        
        print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 初始化检测器
        print("初始化牛只检测器...")
        detector = CattleDetector(confidence=0.25, iou_threshold=0.5)
        
        # 执行检测
        print("执行牛只检测...")
        start_time = time.time()
        results = detector.detect(image)
        elapsed = time.time() - start_time
        
        print(f"检测完成，用时 {elapsed:.4f} 秒")
        print(f"检测到 {len(results)} 头牛")
        
        # 显示检测结果
        for i, result in enumerate(results):
            print(f"牛 #{i+1}:")
            print(f"  - 位置: {[int(x) for x in result['box']]}")
            print(f"  - 置信度: {result['confidence']:.2f}")
            print(f"  - 类别: {result['class_name']}")
        
        # 可视化检测结果
        print("生成可视化结果...")
        vis_image = image.copy()
        
        # 在图像上绘制检测结果
        for i, result in enumerate(results):
            # 获取边界框坐标
            x1, y1, x2, y2 = [int(coord) for coord in result["box"]]
            confidence = result["confidence"]
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"#{i+1} cow: {confidence:.2f}"
            cv2.putText(
                vis_image, 
                label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        # 添加检测统计信息
        cv2.putText(
            vis_image,
            f"检测到 {len(results)} 头牛",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # 保存结果图像
        cv2.imwrite(output_path, vis_image)
        print(f"结果已保存到: {output_path}")
        
        # 保存JSON结果
        json_path = os.path.join(script_dir, "test1_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"JSON结果已保存到: {json_path}")
        
        return True
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print("\n测试" + ("成功" if success else "失败"))
    sys.exit(0 if success else 1)