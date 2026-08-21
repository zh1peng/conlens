# 安装

当前版本从 GitHub 安装：

```bash
git clone https://github.com/zh1peng/conlens.git
cd conlens
python -m pip install .
```

开发环境：

```bash
python -m pip install -e ".[dev]"
pytest --cov=conlens --cov-fail-under=90
```

ConLens 支持 Python 3.10 及以上版本。核心绘图使用 Matplotlib。若要调用可选的 Nilearn
适配器，再安装：

```bash
python -m pip install ".[nilearn]"
```
