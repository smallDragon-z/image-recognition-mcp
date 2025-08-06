import logging
import os
from typing import Optional

from openai import (APIConnectionError, APIError, APITimeoutError, AsyncOpenAI,
                    RateLimitError)

logger = logging.getLogger(__name__)


class OpenAIVision:
    def __init__(self, api_key: Optional[str] = None):
        """初始化OpenAI Vision客户端。

        参数:
            api_key: 可选的API密钥。如果未提供，将尝试从环境中获取。
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未提供OpenAI API密钥且在环境中未找到")

        self.base_url = os.getenv("OPENAI_BASE_URL")
        timeout_value = os.getenv("OPENAI_TIMEOUT", 60)
        self.timeout = float(timeout_value)
        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    async def describe_image(
        self,
        image: str,
        prompt: str = "请详细描述这张图像。",
        mime_type="image/png",
    ) -> str:
        """使用OpenAI的GPT-4 Vision描述图像。

        参数:
            image: 包含base64编码图像的字符串。
            prompt: 包含提示信息的字符串。

        返回:
            str: 图像的描述

        异常:
            Exception: 当API调用失败时抛出异常
        """
        try:
            # 从环境变量获取模型名称，默认使用gpt-4o-mini
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            # 创建消息内容
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=1024,
            )

            # 提取并返回描述
            return response.choices[0].message.content or "未生成有效描述。"

        except APITimeoutError as e:
            logger.error(f"OpenAI API超时: {str(e)}")
            raise Exception(f"请求超时: {str(e)}")
        except APIConnectionError as e:
            logger.error(f"OpenAI API连接错误: {str(e)}")
            raise Exception(f"连接错误: {str(e)}")
        except RateLimitError as e:
            logger.error(f"OpenAI API速率限制超出: {str(e)}")
            raise Exception(f"超出速率限制: {str(e)}")
        except APIError as e:
            logger.error(f"OpenAI API错误: {str(e)}")
            raise Exception(f"API错误: {str(e)}")
        except Exception as e:
            logger.error(f"OpenAI Vision中发生意外错误: {str(e)}", exc_info=True)
            raise Exception(f"意外错误: {str(e)}")