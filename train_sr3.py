# MSML 612 - Step 3
# Train the SR3 diffusion model on the fixed train split in split.json.
# Each batch is (lr 64x64, bicubic condition 256x256, sharp target 256x256, class index).
# Saves the checkpoint with the lowest validation loss to checkpoints/sr3_best.pt
# and the per-epoch losses to results/step3/sr3_training_history.json.
# Use --no-class to train the model without the land-use embedding (Step 6).
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from srcnn import ROOT
from sr3 import SR3
from ucmerced_sr3 import UCMercedTriples

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=Path, default=ROOT / "split.json")
parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "sr3_best.pt")
parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "step3")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--learning-rate", type=float, default=2e-4)
parser.add_argument("--steps", type=int, default=200)  # number of noise steps
parser.add_argument("--base", type=int, default=32)  # UNet width
parser.add_argument("--crop", type=int, default=64)  # training crop size in pixels
parser.add_argument("--no-class", action="store_true")  # turn the class embedding off
parser.add_argument("--seed", type=int, default=612)
parser.add_argument("--device", default=None)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

train_data = UCMercedTriples(args.split, "train")
val_data = UCMercedTriples(args.split, "val")
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size)

use_class = not args.no_class
num_classes = len(train_data.class_names)  # 21 for UC Merced
model = SR3(args.steps, args.base, num_classes, use_class).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
args.results_dir.mkdir(parents=True, exist_ok=True)
history_path = args.results_dir / "sr3_training_history.json"
best_val = float("inf")
history = []
print(f"Device: {device}; train: {len(train_data)}; val: {len(val_data)}; class embedding: {use_class}")

for epoch in range(1, args.epochs + 1):
    model.train()
    train_sum = 0.0
    for lr_img, cond, target, label in train_loader:
        # same random crop from the bicubic and sharp images (much faster than 256x256)
        y = random.randint(0, 256 - args.crop)
        x = random.randint(0, 256 - args.crop)
        # rescale pixel values from [0, 1] to [-1, 1] for diffusion
        cond = cond[:, :, y:y + args.crop, x:x + args.crop].to(device) * 2 - 1
        target = target[:, :, y:y + args.crop, x:x + args.crop].to(device) * 2 - 1
        label = label.to(device)
        optimizer.zero_grad()
        loss = model.loss(cond, target, label)
        loss.backward()
        optimizer.step()
        train_sum += loss.item() * cond.size(0)

    model.eval()
    val_sum = 0.0
    with torch.no_grad():
        for lr_img, cond, target, label in val_loader:
            cond, target, label = cond.to(device) * 2 - 1, target.to(device) * 2 - 1, label.to(device)
            val_sum += model.loss(cond, target, label).item() * cond.size(0)

    train_loss = train_sum / len(train_data)
    val_loss = val_sum / len(val_data)
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    print(f"Epoch {epoch:03d}/{args.epochs}: train loss={train_loss:.5f}, val loss={val_loss:.5f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "steps": args.steps,
                "base": args.base,
                "num_classes": num_classes,
                "use_class": use_class,
                "seed": args.seed,
            },
            args.checkpoint,
        )
        print(f"Saved best checkpoint: {args.checkpoint}")
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
