#!/usr/bin/env python3
"""
测试从URL检测牛只
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 导入MCP服务器中的牛只检测工具
    from src.image_recognition_server.server import detect_cattle_from_file
except ImportError as e:
    print(f"错误: 导入模块失败 - {e}")
    print("请确保您已安装所有必要的依赖并且MCP服务器代码存在")
    sys.exit(1)

async def main():
    """主函数"""
    # 测试URL
    test_url = "https://q1.itc.cn/images01/20250208/a0f5d172263647feac84309336d629aa.jpeg"
    
    print(f"正在测试URL: {test_url}")
    
    try:
        # 调用MCP服务中的牛只检测工具
        print("正在执行牛只检测...")
        result_json = await detect_cattle_from_file(
            filepath=test_url,
            confidence=0.25,
            iou_threshold=0.5
        )
        
        # 解析JSON结果
        result = json.loads(result_json)
        
        # 检查是否有错误
        if "error" in result:
            print(f"检测错误: {result['error']}")
            sys.exit(1)
            
        # 打印检测结果
        print(f"检测到 {result['count']} 头牛:")
        for i, detection in enumerate(result["detections"]):
            print(f"牛 #{i+1}:")
            print(f"  - 位置: {[round(x, 2) for x in detection['box']]}")
            print(f"  - 置信度: {detection['confidence']:.2f}")
            print(f"  - 类别: {detection['class_name']}")
        
        # 将结果保存到文件
        output_path = Path("cattle_url_result.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"错误: 处理图像时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())