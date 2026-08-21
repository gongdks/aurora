## 方式一：源码运行（适合开发者/技术用户）

### 环境要求

- Python 3.11 或更高版本（去 [python.org](https://www.python.org) 下载）
- Windows / macOS / Linux 均可

### 安装步骤

```bash
# 1. 打开命令行，进入项目目录
cd AIAgent

# 2. 创建虚拟环境（推荐，避免依赖冲突）
python -m venv venv

# Windows 激活（二选一）:
#   PowerShell:  先执行 "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"，再运行: venv\Scripts\activate
#   CMD:        直接运行: venv\Scripts\activate.bat
# macOS/Linux 激活:
#   source venv/bin/activate

# 删除虚拟环境 先退出 deactivate
#   PowerShell:  Remove-Item -Recurse -Force venv
#   CMD: rmdir /s /q venv

# 3. 安装依赖（国内用户建议使用清华镜像源加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或永久设置（以后所有 pip 都走镜像）:
#   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#   然后直接: pip install -r requirements.txt

# 4. 配置
copy .env.example .env
# 然后编辑 .env，填入 API Key 等信息

# 5. 启动
python app_qt.py
```