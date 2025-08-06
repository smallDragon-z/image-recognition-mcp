"""牛只检测测试模块"""
import os
import sys
import pytest
from PIL import Image
import base64
import io
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入牛只检测模块
try:
    from src.image_recognition_server.cattle.detector import get_cattle_detector, DEPS_INSTALLED
    from src.image_recognition_server.server import detect_cattle
except ImportError:
    DEPS_INSTALLED = False

# 如果依赖未安装，跳过所有测试
pytestmark = pytest.mark.skipif(
    not DEPS_INSTALLED, reason="牛只检测依赖未安装"
)

@pytest.fixture
def sample_cow_image():
    """创建一个测试图像。"""
    # 这只是一个空白图像，实际测试需要真实的牛图像
    img = Image.new('RGB', (640, 480), color='white')
    
    # 转换为base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return base64_image

@pytest.mark.asyncio
async def test_detect_cattle_empty_image(sample_cow_image):
    """测试牛只检测功能。"""
    if not DEPS_INSTALLED:
        pytest.skip("牛只检测依赖未安装")
    
    # 使用空白图像，应该不会检测到牛
    result = await detect_cattle(image=sample_cow_image)
    result_json = json.loads(result)
    
    assert "count" in result_json
    assert result_json["count"] == 0
    assert "detections" in result_json
    assert isinstance(result_json["detections"], list)

@pytest.mark.asyncio
async def test_detect_cattle_invalid_params():
    """测试无效参数处理。"""
    if not DEPS_INSTALLED:
        pytest.skip("牛只检测依赖未安装")
    
    # 测试没有提供参数
    result = await detect_cattle()
    result_json = json.loads(result)
    assert "error" in result_json
    assert "必须提供" in result_json["error"]
    
    # 测试同时提供两个参数
    result = await detect_cattle(image="invalid", filepath="invalid")
    result_json = json.loads(result)
    assert "error" in result_json
    assert "不能同时提供" in result_json["error"]
    
    # 测试无效的base64数据
    result = await detect_cattle(image="invalid_base64")
    result_json = json.loads(result)
    assert "error" in result_json
    assert "无效的base64图像数据" in result_json["error"]