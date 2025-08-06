import os
import sys
import pytesseract
from PIL import Image

# 打印当前环境变量
print(f"当前TESSERACT_CMD环境变量: {os.getenv('TESSERACT_CMD')}")

# 设置Tesseract路径
tesseract_path = r"C:\Users\74482\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

print(f"设置pytesseract路径为: {pytesseract.pytesseract.tesseract_cmd}")

# 检查文件是否存在
if os.path.exists(tesseract_path):
    print(f"Tesseract可执行文件存在于: {tesseract_path}")
else:
    print(f"错误: Tesseract可执行文件不存在于: {tesseract_path}")
    # 尝试查找可能的位置
    possible_locations = [
        r"C:\Users\74482\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]
    
    for loc in possible_locations:
        if os.path.exists(loc):
            print(f"找到Tesseract在: {loc}")

# 尝试获取Tesseract版本
try:
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract版本: {version}")
except Exception as e:
    print(f"获取Tesseract版本时出错: {str(e)}")

# 尝试简单的OCR测试
try:
    # 创建一个简单的测试图像
    img = Image.new('RGB', (200, 50), color='white')
    # 尝试OCR
    text = pytesseract.image_to_string(img)
    print(f"OCR测试成功! 结果: '{text}'")
except Exception as e:
    print(f"OCR测试失败: {str(e)}")
    print("请确保Tesseract已正确安装，并且可执行文件路径正确。")