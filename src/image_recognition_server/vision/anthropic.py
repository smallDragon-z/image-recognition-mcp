import logging
import os
from typing import Optional

from anthropic import Anthropic, APIConnectionError, APIError, APITimeoutError
from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam

logger = logging.getLogger(__name__)


class AnthropicVision:
    def __init__(self, api_key: Optional[str] = None):
        """初始化Anthropic Vision客户端。

        参数:
            api_key: 可选的API密钥。如果未提供，将尝试从环境中获取。
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "未提供Anthropic API密钥且在环境中未找到"
            )

        self.client = Anthropic(api_key=self.api_key)

    def describe_image(
        self,
        image: str,
        prompt: str = "请详细描述这张图像。",
        mime_type="image/png",
    ) -> str:
        """使用Anthropic的Claude Vision描述图像。

        参数:
            image: 包含base64编码图像的字符串。
            prompt: 可选的提示信息字符串。

        返回:
            str: 生成的图像描述文本

        异常:
            Exception: 当API调用失败时抛出异常
        """
        try:

            image_block = ImageBlockParam(
                type="image",
                source={"type": "base64", "media_type": mime_type, "data": image},
            )

            text_block = TextBlockParam(type="text", text=prompt)

            messages: list[MessageParam] = [
                {
                    "role": "user",
                    "content": [image_block, text_block],
                }
            ]

            # 从环境变量获取模型名称，默认使用claude-3.5-sonnet-beta
            model = os.getenv("ANTHROPIC_MODEL", "claude-3.5-sonnet-beta")

            # 发起API调用
            response = self.client.messages.create(
                model=model, max_tokens=1024, messages=messages
            )

            # 提取响应中的文本内容
            description = []
            for block in response.content:
                if hasattr(block, "text"):
                    description.append(block.text)

            # 返回合并后的描述文本，若无描述则返回默认提示
            if description:
                return " ".join(description)
            return "未生成有效描述。"

        except APITimeoutError as e:
            logger.error(f"Anthropic API请求超时: {str(e)}")
            raise Exception(f"请求超时: {str(e)}")
        except APIConnectionError as e:
            logger.error(f"Anthropic API连接失败: {str(e)}")
            raise Exception(f"连接失败: {str(e)}")
        except APIError as e:
            logger.error(f"Anthropic API返回错误: {str(e)}")
            raise Exception(f"API错误: {str(e)}")
        except Exception as e:
            logger.error(
                f"Anthropic Vision处理过程中发生意外错误: {str(e)}", exc_info=True
            )
            raise Exception(f"处理异常: {str(e)}")