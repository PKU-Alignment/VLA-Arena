#!/usr/bin/env python3
# Copyright 2025 The VLA-Arena Authors.
#
# 批量将 `openpi.` 短路径导入替换为 `vla_arena.models.openpi.src.openpi.` 全路径，
# 修复 ModuleNotFoundError: No module named 'openpi'。
#
# 用法:
#   python fix_openpi_imports.py [--dry-run] [root_dir]
#
# 默认 root_dir 为脚本所在目录的父级 openpi 源码根目录。

import argparse
import json
import os
import re
import sys

# 要替换的模块前缀
OLD_PREFIX = "openpi."
NEW_PREFIX = "vla_arena.models.openpi.src.openpi."

# 仅当 openpi. 不是已存在的全路径一部分时才替换，避免重复替换
# 先合并可能存在的重复前缀（防止多次运行导致 vla_arena.models.vla_arena.models...）
DUPLICATED_PREFIX = "vla_arena.models.vla_arena.models.openpi.src.openpi.src.openpi."
def fix_content(text: str) -> tuple[str, int]:
    """将文本中的 openpi. 替换为全路径（不替换已含全路径的部分），返回 (新文本, 替换次数)。"""
    # 先清理重复前缀
    if DUPLICATED_PREFIX in text:
        text = text.replace(DUPLICATED_PREFIX, NEW_PREFIX)
    # 不替换紧跟在 "vla_arena.models.openpi.src." 后面的 "openpi."
    pattern = r"(?<!vla_arena\.models\.openpi\.src\.)openpi\."
    new_text, n = re.subn(pattern, NEW_PREFIX, text)
    return new_text, n


def fix_py_file(path: str, dry_run: bool) -> int:
    """处理单个 .py 文件，返回替换次数。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    new_content, n = fix_content(content)
    if n == 0:
        return 0
    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    return n


def fix_ipynb_file(path: str, dry_run: bool) -> int:
    """处理单个 .ipynb 文件（替换每个 cell 的 source 中的内容），返回替换次数。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        nb = json.load(f)
    total = 0
    for cell in nb.get("cells", []):
        src = cell.get("source")
        if isinstance(src, list):
            new_src = []
            for line in src:
                new_line, n = fix_content(line)
                new_src.append(new_line)
                total += n
            cell["source"] = new_src
        elif isinstance(src, str):
            new_src, n = fix_content(src)
            cell["source"] = new_src
            total += n
    if total > 0 and not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量将 openpi. 导入替换为 vla_arena.models.openpi.src.openpi."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要修改的文件，不写入",
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=None,
        help="openpi 源码根目录（默认：脚本所在目录的 openpi 根）",
    )
    args = parser.parse_args()

    if args.root_dir is None:
        # 默认：脚本在 openpi 下某处，向上找到 openpi 根
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root = script_dir
        while os.path.basename(root) != "openpi" and os.path.dirname(root) != root:
            root = os.path.dirname(root)
        if os.path.basename(root) != "openpi":
            root = script_dir  # 没找到就用脚本目录
    else:
        root = os.path.abspath(args.root_dir)

    if not os.path.isdir(root):
        print(f"错误: 目录不存在: {root}", file=sys.stderr)
        sys.exit(1)

    total_files = 0
    total_replacements = 0

    this_script = os.path.abspath(__file__)
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.abspath(path) == this_script:
                continue  # 不修改本脚本自身
            if name.endswith(".py"):
                n = fix_py_file(path, args.dry_run)
                if n > 0:
                    total_files += 1
                    total_replacements += n
                    label = "(dry-run) " if args.dry_run else ""
                    print(f"{label}{path}: {n} 处替换")
            elif name.endswith(".ipynb"):
                n = fix_ipynb_file(path, args.dry_run)
                if n > 0:
                    total_files += 1
                    total_replacements += n
                    label = "(dry-run) " if args.dry_run else ""
                    print(f"{label}{path}: {n} 处替换")

    if args.dry_run:
        print(f"\n[dry-run] 共 {total_files} 个文件、{total_replacements} 处将被替换。去掉 --dry-run 执行实际修改。")
    else:
        print(f"\n完成: {total_files} 个文件、{total_replacements} 处已替换。")


if __name__ == "__main__":
    main()
