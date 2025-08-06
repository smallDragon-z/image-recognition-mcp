import base64
import io
import json
import logging
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from PIL import Image

from .utils.image import image_to_base64, validate_base64_image
from .utils.ocr import OCRError, extract_text_from_image

# 加载环境变量
load_dotenv()

# 配置编码，默认为UTF-8
DEFAULT_ENCODING = "utf-8"
ENCODING = os.getenv("MCP_OUTPUT_ENCODING", DEFAULT_ENCODING)

# 配置日志记录到文件
log_file_path = os.path.join(os.path.dirname(__file__), "mcp_server.log")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=log_file_path,
    filemode="a",  # 追加到日志文件
)
logger = logging.getLogger(__name__)

logger.info(f"使用编码: {ENCODING}")

# 导入牛只检测模块（如果依赖已安装）
try:
    from .cattle.detector import get_cattle_detector, DEPS_INSTALLED as CATTLE_DEPS_INSTALLED
except ImportError:
    logger.warning("缺少牛只检测依赖，detect_cattle功能将不可用")
    CATTLE_DEPS_INSTALLED = False


def sanitize_output(text: str) -> str:
    """清理输出字符串，替换有问题的字符。"""
    if text is None:
        return ""  # 对于None返回空字符串
    try:
        return text.encode(ENCODING, "replace").decode(ENCODING)
    except Exception as e:
        logger.error(f"清理过程中出错: {str(e)}", exc_info=True)
        return text  # 如果清理失败，返回原始文本


# 创建MCP服务器
mcp = FastMCP("mcp-image-recognition")





async def process_image_with_ocr(image_data: str) -> str:
    """使用OCR处理图像，提取文本内容。

    参数:
        image_data: Base64编码的图像数据

    返回:
        str: 从图像中提取的文本
    """
    try:
        # 将base64转换为PIL图像
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # 使用OCR提取文本
        ocr_text = extract_text_from_image(image, ocr_required=True)
        
        if not ocr_text:
            return "未从图像中检测到任何文本。"
        
        return sanitize_output(ocr_text)
    except OCRError as e:
        logger.error(f"OCR处理失败: {str(e)}")
        raise ValueError(f"OCR错误: {str(e)}")
    except Exception as e:
        logger.error(f"OCR过程中发生意外错误: {str(e)}")
        raise


@mcp.tool()
async def describe_image(
    image: str
) -> str:
    """从图像中提取文本内容。

    参数:
        image: Base64编码的图像数据

    返回:
        str: 从图像中提取的文本
    """
    try:
        logger.info("处理OCR文本提取请求")
        logger.debug(f"图像数据长度: {len(image)}")

        # 验证图像数据
        if not validate_base64_image(image):
            raise ValueError("无效的base64图像数据")

        result = await process_image_with_ocr(image)
        if not result:
            return "未从图像中检测到任何文本。"

        logger.info("成功从图像中提取文本")
        return result
    except ValueError as e:
        logger.error(f"输入错误: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"提取文本时出错: {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def describe_image_from_url(
    url: str
) -> str:
    """从URL图像中提取文本内容。

    参数:
        url: 图像的URL地址

    返回:
        str: 从图像中提取的文本
    """
    try:
        # 检查URL是否有效
        if not url.startswith(('http://', 'https://')):
            raise ValueError("无效的URL地址，必须以http://或https://开头")
            
        logger.info(f"从URL加载图像: {url}")
        
        # 从URL下载图像
        import requests
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image_data = base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            raise ValueError(f"无法从URL加载图像: {str(e)}")
            
        logger.info("成功从URL下载图像")
        logger.debug(f"Base64数据长度: {len(image_data)}")

        # 使用describe_image工具
        result = await describe_image(image=image_data)

        if not result:
            return "未从图像中检测到任何文本。"

        return result
    except ValueError as e:
        logger.error(f"输入错误: {str(e)}")
        raise
    except requests.RequestException as e:
        logger.error(f"URL请求错误: {str(e)}")
        raise ValueError(f"URL请求错误: {str(e)}")
    except Exception as e:
        logger.error(f"处理URL图像时出错: {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def detect_cattle(
    url: str, confidence: float = 0.25, iou_threshold: float = 0.5
) -> str:
    """从URL检测图像中的牛只。

    参数:
        url: 图像的URL地址
        confidence: 检测置信度阈值，默认0.25
        iou_threshold: NMS的IoU阈值，默认0.5

    返回:
        str: JSON格式的检测结果
    """
    if not CATTLE_DEPS_INSTALLED:
        return sanitize_output(json.dumps({
            "error": "缺少依赖项: supervision 或 ultralytics。请安装依赖: pip install supervision ultralytics opencv-python"
        }, ensure_ascii=False))

    try:
        # 检查URL是否有效
        if not url.startswith(('http://', 'https://')):
            raise ValueError("无效的URL地址，必须以http://或https://开头")

        # 从URL加载图像
        logger.info(f"从URL加载图像: {url}")
        import requests
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image_data = base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            raise ValueError(f"无法从URL加载图像: {str(e)}")

        # 获取牛只检测器实例
        detector = get_cattle_detector()

        # 解码图像
        image_bytes = base64.b64decode(image_data)
        import cv2
        import numpy as np
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 执行检测
        logger.info(f"执行牛只检测，置信度阈值: {confidence}, IoU阈值: {iou_threshold}")
        results = detector.detect(img, confidence=confidence, iou_threshold=iou_threshold)

        # 返回JSON格式的结果
        result_json = {
            "count": len(results),
            "detections": results
        }
        
        logger.info(f"检测到 {len(results)} 头牛")
        return sanitize_output(json.dumps(result_json, ensure_ascii=False))

    except requests.RequestException as e:
        logger.error(f"URL请求错误: {str(e)}")
        return sanitize_output(json.dumps({"error": f"URL请求错误: {str(e)}"}, ensure_ascii=False))
    except ValueError as e:
        logger.error(f"输入错误: {str(e)}")
        return sanitize_output(json.dumps({"error": f"输入错误: {str(e)}"}, ensure_ascii=False))
    except Exception as e:
        logger.error(f"检测牛只时出错: {str(e)}", exc_info=True)
        return sanitize_output(json.dumps({"error": f"检测牛只时出错: {str(e)}"}, ensure_ascii=False))




if __name__ == "__main__":
    mcp.run()