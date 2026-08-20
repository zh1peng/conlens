---
title: 安装
description: 安装 ConLens、可选依赖与开发环境
---

# 安装

ConLens 支持 Python 3.10 及以上版本，可在 Linux、macOS 和 Windows 上运行。

## 从公开 GitHub 仓库安装

```bash
python -m pip install "conlens @ git+https://github.com/zh1peng/conlens.git"
```

也可以克隆仓库后安装：

```bash
git clone https://github.com/zh1peng/conlens.git
cd conlens
python -m pip install .
```

## 可选的 Nilearn 集成

```bash
python -m pip install ".[nilearn]"
```

核心包不会自动导入 Nilearn；只有调用 `conlens.interfaces.nilearn` 中的适配函数时才需要该依赖。

## 验证安装

```bash
python -c "import conlens; print(conlens.__version__)"
```

## 开发环境

```bash
git clone https://github.com/zh1peng/conlens.git
cd conlens
python -m pip install -e ".[dev]"
pytest --cov=conlens
ruff check .
```

::: tip 文档开发
VitePress 站点位于 `website/`。进入该目录后运行 `npm install` 和 `npm run docs:dev` 即可启动本地文档服务器。
:::

