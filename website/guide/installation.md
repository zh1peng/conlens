# 安装与检查

ConLens 需要 Python 3.10 或以上版本。以下从仓库安装，同时获得教程脚本：

```bash
git clone https://github.com/zh1peng/conlens.git
cd conlens
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux 激活环境：

```bash
source .venv/bin/activate
```

然后安装并确认当前解释器可以导入：

```bash
python -m pip install .
python -c "import conlens; print(conlens.__version__)"
python -m examples.teaching_workflow
```

最后一条运行完整模拟示例并生成结果，见[运行第一个分析](/guide/quick-start)。导入失败时，核对运行安装和脚本的 Python 是否同一个虚拟环境解释器。

核心绘图使用 Matplotlib；仅需 Nilearn 适配器时再安装：

```bash
python -m pip install ".[nilearn]"
```

开发者在仓库根目录使用 `python -m pip install -e ".[dev]"`，再执行 `pytest --cov=conlens --cov-fail-under=90`。这些软件测试与统计校准的作用不同。
