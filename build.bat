@echo off
REM MCP图像识别服务器的构建脚本

REM 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

REM 运行代码格式化
black src/ 
isort src/ 

REM 运行代码检查
ruff check src/
mypy src/ 

REM 构建包
python setup.py build






REM 运行代码格式化
@REM black tests/
@REM isort tests/

REM 运行代码检查
@REM ruff check tests/
@REM mypy tests/

REM 运行测试
@REM pytest tests/ -v --cov=src