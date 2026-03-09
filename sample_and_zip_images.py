import os, random, zipfile, argparse, shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True, help="root folder with class subfolders")
parser.add_argument("--out", default="sampled_images.zip")
parser.add_argument("--per_class", type=int, default=500, help="max images per class")
args = parser.parse_args()

tmp = Path("tmp_images_to_zip")
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True, exist_ok=True)

for cls in sorted(os.listdir(args.src)):
    cdir = Path(args.src)/cls
    if not cdir.is_dir():
        continue
    imgs = [p for p in cdir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.shuffle(imgs)
    keep = imgs[:args.per_class]
    for p in keep:
        dest = tmp/cls/p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)

with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in tmp.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(tmp))
print(f"Wrote {args.out}")
