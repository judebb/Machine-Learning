# Step 3
# Score the trained SR3 model on a split (val by default) using the same
# PSNR/SSIM code (metrics.py) as the bicubic and SRCNN evaluation.
# Writes a JSON summary and a per-image CSV under results/step3/
# Use --name sr3_noclass (and the no-class checkpoint) for the Step 6 model so files don't overwrite.

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

from metrics import image_metrics, summarize_metrics
from srcnn import ROOT
from sr3 import SR3
from ucmerced_sr3 import UCMercedTriples

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=Path, default=ROOT / "split.json")
parser.add_argument("--split-name", default="val", choices=("val", "test"))
parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "sr3_best.pt")
parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "step3")
parser.add_argument("--name", default="sr3")  # prefix for output files
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--ddim-steps", type=int, default=50)  # number of DDIM sampling steps
parser.add_argument("--limit", type=int, default=None, help="score only the first N images")
parser.add_argument("--save-every", type=int, default=16)  # save one picture every N images
parser.add_argument("--seed", type=int, default=612)
parser.add_argument("--device", default=None)
args = parser.parse_args()

# sampling uses random noise, so the seed makes the results repeatable
torch.manual_seed(args.seed)
device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

# build an empty SR3 with the same settings used in training, then load the saved weights
checkpoint = torch.load(args.checkpoint, map_location=device)
model = SR3(checkpoint["steps"], checkpoint["base"], checkpoint["num_classes"], checkpoint["use_class"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dataset = UCMercedTriples(args.split, args.split_name)
count = min(args.limit or len(dataset), len(dataset))

samples_dir = args.results_dir / f"{args.name}_samples"
samples_dir.mkdir(parents=True, exist_ok=True)

rows = []
csv_path = args.results_dir / f"{args.name}_{args.split_name}_per_image.csv"

for start in range(0, count, args.batch_size):
    idx = list(range(start, min(start + args.batch_size, count)))
    items = [dataset[i] for i in idx]  # each item: (64x64 input, bicubic, target, class)
    cond = torch.stack([p[1] for p in items]).to(device) * 2 - 1  # bicubic images, scaled to [-1, 1]
    target = torch.stack([p[2] for p in items])  # sharp originals, kept in [0, 1]
    label = torch.stack([p[3] for p in items]).to(device)
    # generate sharpened images from noise with DDIM, then scale back to [0, 1]
    pred = ((model.sample(cond, label, args.ddim_steps).cpu() + 1) / 2).clamp(0, 1)

    # saves a side-by-side picture: bicubic | SR3 | original
    for i, c, t, p in zip(idx, cond.cpu(), target, pred):
        if i % args.save_every == 0:
            strip = torch.cat([(c + 1) / 2, p, t], dim=2)  # dim=2 joins them left to right
            array = (strip.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            Image.fromarray(array).save(samples_dir / f"{i:03d}.png")

    for i, t, p in zip(idx, target, pred):
        rows.append(
            {
                "image": dataset.paths[i],
                **image_metrics(t.permute(1, 2, 0).numpy(), p.permute(1, 2, 0).numpy()),
            }
        )
    print(f"Scored {len(rows)}/{count} images")

    # save scores to csv after each batch
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "psnr", "ssim"))
        writer.writeheader()
        writer.writerows(rows)

# average PSNR/SSIM over all scored images and save the results
summary = summarize_metrics(rows)
metrics_path = args.results_dir / f"{args.name}_{args.split_name}_metrics.json"
payload = {
    "method": args.name,
    "split": args.split_name,
    "image_count": len(rows),
    "class_embedding": checkpoint["use_class"],
    "training_steps": checkpoint["steps"],
    "sampling_steps": args.ddim_steps,
    "checkpoint": str(args.checkpoint.resolve()),
    "metrics": summary,
}
metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

print(f"PSNR: {summary['psnr']['mean']:.4f} +/- {summary['psnr']['std']:.4f} dB")
print(f"SSIM: {summary['ssim']['mean']:.6f} +/- {summary['ssim']['std']:.6f}")
print(f"Summary: {metrics_path}")
print(f"Per-image scores: {csv_path}")
