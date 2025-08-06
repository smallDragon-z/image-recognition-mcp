import os
import pytest
from PIL import Image, ImageDraw, ImageFont
from src.image_recognition_server.utils.ocr import extract_text_from_image, OCRError

@pytest.fixture
def text_image():
    """创建一个带文本的测试图像。"""
    # 创建一个更大对比度的图像
    img = Image.new('RGB', (800, 200), color='white')
    d = ImageDraw.Draw(img)
    
    # 创建一个更易于OCR的简单测试字符串
    test_string = "TEST"
    
    # 以大而清晰的字体绘制文本
    d.text((100, 50), test_string, fill='black', font=None)
    return img, test_string


@pytest.fixture
def empty_image():
    """创建一个空白的测试图像。"""
    return Image.new('RGB', (100, 100), color='white')

def test_basic_text_extraction(text_image):
    """测试从带有清晰文本的图像中提取文本。"""
    img, expected_text = text_image
    result = extract_text_from_image(img)
    assert result is not None
    assert expected_text in result.upper()  # 转换为大写进行比较

def test_empty_image(empty_image):
    """测试处理无文本图像。"""
    result = extract_text_from_image(empty_image)
    assert result is None

def test_tesseract_not_available(monkeypatch):
    """测试Tesseract不可访问时的错误处理。"""
    # 创建一个简单的测试图像
    img = Image.new('RGB', (100, 100), color='white')
    
    # 模拟pytesseract抛出错误
    def mock_image_to_string(*args, **kwargs):
        raise Exception("tesseract is not installed or it's not in your PATH")
    
    monkeypatch.setattr("pytesseract.image_to_string", mock_image_to_string)
    
    # 使用ocr_required=False进行测试
    result = extract_text_from_image(img, ocr_required=False)
    assert result is None
    
    # 使用ocr_required=True进行测试
    with pytest.raises(OCRError) as exc_info:
        extract_text_from_image(img, ocr_required=True)
    assert "Tesseract OCR未安装或不在PATH中" in str(exc_info.value)

def test_custom_tesseract_path(monkeypatch):
    """测试使用环境变量中的自定义Tesseract路径。"""
    custom_path = r"D:\ocr\tesseract.exe"
    
    # 模拟环境变量
    monkeypatch.setenv("TESSERACT_CMD", custom_path)
    
    # 模拟pytesseract以验证自定义路径已设置
    def mock_image_to_string(*args, **kwargs):
        import pytesseract
        assert pytesseract.pytesseract.tesseract_cmd == custom_path
        return "Hello World"
    
    monkeypatch.setattr("pytesseract.image_to_string", mock_image_to_string)
    
    # 创建一个简单的测试图像
    img = Image.new('RGB', (100, 100), color='white')
    result = extract_text_from_image(img)
    assert result == "Hello World"

def test_ocr_required_flag(monkeypatch):
    """测试ocr_required标志的True/False行为。"""
    img = Image.new('RGB', (100, 100), color='white')
    
    def mock_image_to_string(*args, **kwargs):
        return ""  # 模拟未找到文本
    
    monkeypatch.setattr("pytesseract.image_to_string", mock_image_to_string)
    
    # 使用ocr_required=False（默认）
    result = extract_text_from_image(img)
    assert result is None
    
    # 使用ocr_required=True
    result = extract_text_from_image(img, ocr_required=True)
    assert result is None  # 应该仍然为None，因为空字符串被转换为None