"""
preprocess_retinal.py

将 DRIVE / STARE 数据集预处理为 VM-UNet 兼容格式：
  - 所有图像统一转为 PNG
  - DRIVE：去掉文件名中的 _training / _test / _manual1，使 image 与 mask 同名
      21_training.tif  →  21.png
      01_test.tif      →  01.png
      21_manual1.gif   →  21.png
      01_manual1.gif   →  01.png
  - STARE：取第一个点前的 stem，使 image 与 mask 同名
      im0001.ppm       →  im0001.png
      im0001.ah.ppm    →  im0001.png

用法（在项目根目录执行）：
    python datasets/preprocess_retinal.py
    python datasets/preprocess_retinal.py --data_root ./data --datasets DRIVE STARE
    python datasets/preprocess_retinal.py --keep_orig   # 保留原始文件
"""

import argparse
import re
from pathlib import Path

from PIL import Image


# ------------------------------------------------------------------ DRIVE ---

_DRIVE_REMOVE = re.compile(r'_(training|test|manual\d*)', re.IGNORECASE)


def _drive_new_stem(stem: str) -> str:
    """21_training → 21，01_manual1 → 01"""
    return _DRIVE_REMOVE.sub('', stem).strip('_')


def process_dir(src_dir: Path, new_stem_fn, is_mask: bool, remove_orig: bool):
    """遍历目录，把每个文件转换成 PNG 并重命名。"""
    if not src_dir.exists():
        print(f"  [SKIP] 目录不存在：{src_dir}")
        return

    files = sorted(f for f in src_dir.iterdir() if f.is_file())
    print(f"  {src_dir.relative_to(src_dir.parents[3])}  ({len(files)} 个文件)")

    mode = 'L' if is_mask else 'RGB'
    for f in files:
        new_stem = new_stem_fn(f)
        dst = src_dir / f"{new_stem}.png"

        try:
            img = Image.open(f).convert(mode)
            img.save(dst)
        except Exception as e:
            print(f"    [ERROR] {f.name}: {e}")
            continue

        if f != dst:
            print(f"    {f.name}  →  {dst.name}")
            if remove_orig:
                f.unlink()
        else:
            print(f"    {f.name}  (格式已转换，文件名不变)")


def convert_drive(data_dir: Path, remove_orig: bool):
    print(f"\n{'='*10} DRIVE {'='*10}")
    for split in ('train', 'val'):
        for kind, is_mask in (('images', False), ('masks', True)):
            process_dir(
                src_dir=data_dir / split / kind,
                new_stem_fn=lambda f: _drive_new_stem(f.stem),
                is_mask=is_mask,
                remove_orig=remove_orig,
            )


# ------------------------------------------------------------------ STARE ---

def _stare_new_stem(f: Path) -> str:
    """im0001.ppm → im0001，im0001.ah.ppm → im0001（取第一个点前的部分）"""
    return f.name.split('.')[0]


def convert_stare(data_dir: Path, remove_orig: bool):
    print(f"\n{'='*10} STARE {'='*10}")
    for split in ('train', 'val'):
        for kind, is_mask in (('images', False), ('masks', True)):
            process_dir(
                src_dir=data_dir / split / kind,
                new_stem_fn=_stare_new_stem,
                is_mask=is_mask,
                remove_orig=remove_orig,
            )


# ------------------------------------------------------------------- main ---

def main():
    parser = argparse.ArgumentParser(
        description='将 DRIVE / STARE 转换为 VM-UNet 兼容的 PNG 格式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data_root', default='./data',
        help='数据根目录（包含 DRIVE/ 和 STARE/ 子目录）',
    )
    parser.add_argument(
        '--datasets', nargs='+', default=['DRIVE', 'STARE'],
        choices=['DRIVE', 'STARE'],
        help='要处理的数据集',
    )
    parser.add_argument(
        '--keep_orig', action='store_true',
        help='保留原始文件（默认转换完成后删除原始文件）',
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    remove_orig = not args.keep_orig

    if 'DRIVE' in args.datasets:
        drive_dir = data_root / 'DRIVE'
        if drive_dir.exists():
            convert_drive(drive_dir, remove_orig)
        else:
            print(f"[WARN] DRIVE 目录不存在：{drive_dir}")

    if 'STARE' in args.datasets:
        stare_dir = data_root / 'STARE'
        if stare_dir.exists():
            convert_stare(stare_dir, remove_orig)
        else:
            print(f"[WARN] STARE 目录不存在：{stare_dir}")

    print("\n预处理完成！")
    print("处理后目录结构示例：")
    print("  data/DRIVE/train/images/21.png  ↔  data/DRIVE/train/masks/21.png")
    print("  data/STARE/val/images/im0162.png  ↔  data/STARE/val/masks/im0162.png")


if __name__ == '__main__':
    main()
