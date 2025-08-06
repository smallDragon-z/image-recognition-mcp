import logging
import os
from typing import Optional

import pytesseract  # type: ignore
from PIL import Image

logger = logging.getLogger(__name__)


class OCRError(Exception):
    """OCR相关错误的异常。"""

    pass


def extract_text_from_image(
    image: Image.Image, ocr_required: bool = False
) -> Optional[str]:
    """使用Tesseract OCR从图像中提取文本。

    参数:
        image: 要处理的PIL图像对象
        ocr_required: 如果为True，OCR失败时抛出错误。如果为False，返回None。

    返回:
        Optional[str]: 如果成功则返回提取的文本，如果Tesseract不可用
                      且ocr_required为False则返回None

    异常:
        OCRError: 如果OCR失败且ocr_required为True
    """
    try:
        # 检查环境中是否设置了自定义tesseract路径且不为空
        if tesseract_cmd := os.getenv("TESSERACT_CMD"):
            if tesseract_cmd.strip():  # 仅当路径非空时设置
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # 从图像中提取文本
        text = pytesseract.image_to_string(image)

        # 清理并验证结果
        text = text.strip()
        if text:
            logger.info("使用Tesseract成功从图像中提取文本")
            logger.debug(f"提取的文本长度: {len(text)}")
            return text
        else:
            logger.info("图像中未找到文本")
            return None

    except Exception as e:
        error_msg = f"使用Tesseract提取文本失败: {str(e)}"
        if "not installed" in str(e) or "not in your PATH" in str(e):
            error_msg = (
                "Tesseract OCR未安装或不在PATH中。"
                "请安装Tesseract并确保它在系统PATH中，"
                "或设置TESSERACT_CMD环境变量为可执行文件路径。"
            )

        logger.warning(error_msg)
        if ocr_required:
            raise OCRError(error_msg)
        return None