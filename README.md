# MCP 图像识别服务器

一个提供图像识别功能的MCP服务器，使用Anthropic和OpenAI视觉API。版本0.1.2。

## 功能

- 使用Anthropic Claude Vision或OpenAI GPT-4 Vision进行图像描述
- 支持多种图像格式（JPEG、PNG、GIF、WebP）
- 可配置主要和备用提供商
- 支持Base64和基于文件的图像输入
- 可选的使用Tesseract OCR进行文本提取

## 要求

- Python 3.8或更高版本
- Tesseract OCR（可选）- 文本提取功能需要
  - Windows：从[UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)下载并安装
  - Linux：`sudo apt-get install tesseract-ocr`
  - macOS：`brew install tesseract`

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/mario-andreschak/mcp-image-recognition.git
cd mcp-image-recognition
```

2. 创建并配置环境文件：
```bash
cp .env.example .env
# 使用你的API密钥和首选项编辑.env
```

3. 构建项目：
```bash
build.bat
```

## 使用方法

### 运行服务器
使用python启动服务器：
```bash
python -m image_recognition_server.server
```

使用批处理文件启动服务器：
```bash
run.bat server
```

在开发模式下使用MCP检查器启动服务器：
```bash
run.bat debug
```

### 可用工具

1. `describe_image`
   - 输入：Base64编码的图像数据和MIME类型
   - 输出：图像的详细描述

2. `describe_image_from_file`
   - 输入：图像文件的路径
   - 输出：图像的详细描述

### 环境配置

- `ANTHROPIC_API_KEY`：你的Anthropic API密钥。
- `OPENAI_API_KEY`：你的OpenAI API密钥。
- `VISION_PROVIDER`：主要视觉提供商（`anthropic`或`openai`）。
- `FALLBACK_PROVIDER`：可选的备用提供商。
- `LOG_LEVEL`：日志级别（DEBUG、INFO、WARNING、ERROR）。
- `ENABLE_OCR`：启用Tesseract OCR文本提取（`true`或`false`）。
- `TESSERACT_CMD`：可选的Tesseract可执行文件自定义路径。
- `OPENAI_MODEL`：OpenAI模型（默认：`gpt-4o-mini`）。可以使用OpenRouter格式用于其他模型（例如，`anthropic/claude-3.5-sonnet:beta`）。
- `OPENAI_BASE_URL`：OpenAI API的可选自定义基础URL。设置为`https://openrouter.ai/api/v1`以使用OpenRouter。
- `OPENAI_TIMEOUT`：OpenAI API的可选自定义超时（以秒为单位）。

### 使用OpenRouter

OpenRouter允许你使用OpenAI API格式访问各种模型。要使用OpenRouter，请按照以下步骤操作：

1. 从OpenRouter获取OpenAI API密钥。
2. 在`.env`文件中将`OPENAI_API_KEY`设置为你的OpenRouter API密钥。
3. 将`OPENAI_BASE_URL`设置为`https://openrouter.ai/api/v1`。
4. 将`OPENAI_MODEL`设置为使用OpenRouter格式的所需模型（例如，`anthropic/claude-3.5-sonnet:beta`）。
5. 将`VISION_PROVIDER`设置为`openai`。

### 默认模型

- Anthropic：`claude-3.5-sonnet-beta`
- OpenAI：`gpt-4o-mini`
- OpenRouter：在`OPENAI_MODEL`中使用`anthropic/claude-3.5-sonnet:beta`格式。

## 开发

### 运行测试

运行所有测试：
```bash
run.bat test
```

运行特定测试套件：
```bash
run.bat test server
run.bat test anthropic
run.bat test openai
```

### Docker支持

构建Docker镜像：
```bash
docker build -t mcp-image-recognition .
```

运行容器：
```bash
docker run -it --env-file .env mcp-image-recognition
```

## 许可证

MIT许可证 - 详情请参阅LICENSE文件。

## 发布历史

- **0.1.2**（2025-02-20）：改进了OCR错误处理并为OCR功能添加了全面的测试覆盖
- **0.1.1**（2025-02-19）：添加了Tesseract OCR支持，用于从图像中提取文本（可选功能）
- **0.1.0**（2025-02-19）：初始版本，支持Anthropic和OpenAI视觉功能