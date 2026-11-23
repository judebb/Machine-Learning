# MSML 612 - Step 3 check: one forward + backward pass on a UC Merced batch,
# first with the class embedding on, then the same batch with it off.
import torch
from torch.utils.data import DataLoader

from srcnn import ROOT
from sr3 import SR3
from ucmerced_sr3 import UCMercedTriples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = UCMercedTriples(ROOT / "split.json", "train")
lr_img, cond, target, label = next(iter(DataLoader(data, batch_size=4, shuffle=True)))
print(len(data.class_names), "classes")
print("lr", tuple(lr_img.shape), "| bicubic cond", tuple(cond.shape), "| target", tuple(target.shape), "| class", label.tolist())

cond, target, label = cond.to(device) * 2 - 1, target.to(device) * 2 - 1, label.to(device)
for use_class in (True, False):
    model = SR3(200, 32, len(data.class_names), use_class).to(device)
    loss = model.loss(cond, target, label)
    loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"class embedding {'on ' if use_class else 'off'}: loss={loss.item():.4f}, grad sum={grad:.3f}, "
          f"embedding got gradient: {model.net.cemb.weight.grad is not None}")

out = model.sample(cond[:1], label[:1], ddim_steps=10)
print("DDIM sample shape", tuple(out.shape))
