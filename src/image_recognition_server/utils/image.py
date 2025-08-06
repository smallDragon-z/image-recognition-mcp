import base64
import io
import logging
from pathlib import Path
from typing import Tuple

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


def image_to_base64(image_path: str) -> Tuple[str, str]:
    """将图像文件转换为base64字符串并检测其MIME类型。

    参数:
        image_path: 图像文件的路径

    返回:
        包含(base64_string, mime_type)的元组

    异常:
        FileNotFoundError: 如果图像文件不存在
        ValueError: 如果文件不是有效的图像
    """
    path = Path(image_path)
    if not path.exists():
        logger.error(f"找不到图像文件: {image_path}")
        raise FileNotFoundError(f"找不到图像文件: {image_path}")

    try:
        # 尝试打开并验证图像
        with Image.open(path) as img:
            # 获取图像格式并转换为MIME类型
            format_to_mime = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "GIF": "image/gif",
                "WEBP": "image/webp",
            }
            mime_type = format_to_mime.get(img.format, "application/octet-stream")
            logger.info(
                f"正在处理图像: {image_path}, 格式: {img.format}, 大小: {img.size}"
            )

            # 转换为base64
            with path.open("rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")
                logger.debug(f"Base64数据长度: {len(base64_data)}")

            return base64_data, mime_type

    except UnidentifiedImageError as e:
        logger.error(f"无效的图像格式: {str(e)}")
        raise ValueError(f"无效的图像格式: {str(e)}")
    except OSError as e:
        logger.error(f"读取图像文件失败: {str(e)}")
        raise ValueError(f"读取图像文件失败: {str(e)}")
    except Exception as e:
        logger.error(f"处理图像时发生意外错误: {str(e)}", exc_info=True)
        raise ValueError(f"图像处理失败: {str(e)}")


def validate_base64_image(base64_string: str) -> bool:
    """验证字符串是否为有效的base64编码图像。

    参数:
        base64_string: 要验证的base64字符串

    返回:
        如果有效返回True，否则返回False
    """
    try:
        # 尝试解码base64
        image_data = base64.b64decode(base64_string)

        # 尝试作为图像打开
        with Image.open(io.BytesIO(image_data)) as img:
            logger.debug(
                f"验证base64图像, 格式: {img.format}, 大小: {img.size}"
            )
            return True

    except Exception as e:
        logger.warning(f"无效的base64图像: {str(e)}")
        return False