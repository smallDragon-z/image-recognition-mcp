import base64
import io
import json
import logging
import os
from typing import Union, List, Dict, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from PIL import Image

from .utils.image import image_to_base64, validate_base64_image
from .utils.ocr import OCRError, extract_text_from_image
from .vision.anthropic import AnthropicVision
from .vision.openai import OpenAIVision

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


# 初始化视觉客户端
def get_vision_client() -> Union[AnthropicVision, OpenAIVision]:
    """根据环境设置获取配置的视觉客户端。"""
    provider = os.getenv("VISION_PROVIDER", "anthropic").lower()

    try:
        if provider == "anthropic":
            return AnthropicVision()
        elif provider == "openai":
            return OpenAIVision()
        else:
            raise ValueError(f"无效的视觉提供商: {provider}")
    except Exception as e:
        # 如果配置了备用提供商，则尝试使用
        fallback = os.getenv("FALLBACK_PROVIDER")
        if fallback and fallback.lower() != provider:
            logger.warning(
                f"主要提供商失败: {str(e)}。尝试备用提供商: {fallback}"
            )
            if fallback.lower() == "anthropic":
                return AnthropicVision()
            elif fallback.lower() == "openai":
                return OpenAIVision()
        raise


async def process_image_with_ocr(image_data: str, prompt: str) -> str:
    """使用视觉AI和OCR处理图像。

    参数:
        image_data: Base64编码的图像数据
        prompt: 视觉AI的提示

    返回:
        str: 来自视觉AI和OCR的组合描述
    """
    # 获取视觉AI描述
    client = get_vision_client()

    # 处理同步(Anthropic)和异步(OpenAI)客户端
    if isinstance(client, OpenAIVision):
        description = await client.describe_image(image_data, prompt)
    else:
        description = client.describe_image(image_data, prompt)

    # 检查空或默认响应
    if not description or description == "No description available.":
        raise ValueError("视觉API返回空或默认响应")

    # 如果启用了OCR，则处理OCR
    ocr_enabled = os.getenv("ENABLE_OCR", "false").lower() == "true"
    if ocr_enabled:
        try:
            # 将base64转换为PIL图像
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            # 使用OCR必需标志提取文本
            if ocr_text := extract_text_from_image(image, ocr_required=True):
                description += (
                    f"\n\n此外，这是tesseract-ocr的输出结果: {ocr_text}"
                )
        except OCRError as e:
            # 当OCR启用时传播OCR错误
            logger.error(f"OCR处理失败: {str(e)}")
            raise ValueError(f"OCR错误: {str(e)}")
        except Exception as e:
            logger.error(f"OCR过程中发生意外错误: {str(e)}")
            raise

    return sanitize_output(description)


@mcp.tool()
async def describe_image(
    image: str, prompt: str = "请详细描述这张图像。"
) -> str:
    """使用视觉AI描述图像内容。

    参数:
        image: 图像数据和MIME类型
        prompt: 用于描述的可选提示。

    返回:
        str: 图像的详细描述
    """
    try:
        logger.info(f"处理图像描述请求，提示为: {prompt}")
        logger.debug(f"图像数据长度: {len(image)}")

        # 验证图像数据
        if not validate_base64_image(image):
            raise ValueError("无效的base64图像数据")

        result = await process_image_with_ocr(image, prompt)
        if not result:
            raise ValueError("处理过程收到空响应")

        logger.info("成功处理图像")
        return sanitize_output(result)
    except ValueError as e:
        logger.error(f"输入错误: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"描述图像时出错: {str(e)}", exc_info=True)
        raise


@mcp.tool()
async def describe_image_from_file(
    filepath: str, prompt: str = "请详细描述这张图像。"
) -> str:
    """使用视觉AI描述图像文件的内容。

    参数:
        filepath: 图像文件的路径
        prompt: 用于描述的可选提示。

    返回:
        str: 图像的详细描述
    """
    try:
        logger.info(f"处理图像文件: {filepath}")

        # 将图像转换为base64
        image_data, mime_type = image_to_base64(filepath)
        logger.info(f"成功将图像转换为base64。MIME类型: {mime_type}")
        logger.debug(f"Base64数据长度: {len(image_data)}")

        # 使用describe_image工具
        result = await describe_image(image=image_data, prompt=prompt)

        if not result:
            raise ValueError("处理过程收到空响应")

        return sanitize_output(result)
    except FileNotFoundError:
        logger.error(f"未找到图像文件: {filepath}")
        raise
    except ValueError as e:
        logger.error(f"输入错误: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"处理图像文件时出错: {str(e)}", exc_info=True)
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


@mcp.tool()
async def detect_cattle_from_file(
    filepath: str, confidence: float = 0.25, iou_threshold: float = 0.5
) -> str:
    """已弃用，请使用detect_cattle工具。

    参数:
        filepath: 图像文件路径或URL
        confidence: 检测置信度阈值，默认0.25
        iou_threshold: NMS的IoU阈值，默认0.5

    返回:
        str: JSON格式的检测结果
    """
    # 如果是URL，调用detect_cattle
    if filepath.startswith(('http://', 'https://')):
        return await detect_cattle(url=filepath, confidence=confidence, iou_threshold=iou_threshold)
    else:
        return json.dumps({
            "error": "该工具已不再支持本地文件路径。请使用detect_cattle工具并提供URL地址。"
        }, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()