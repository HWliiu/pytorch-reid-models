"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

from pathlib import Path

import accelerate
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from reid_models.modeling import _build_reid_model
from reid_models.utils import set_seed, setup_logger


def train(accelerator, model, train_loader, optimizer, criterion, max_epoch, epoch):
    model.train()
    bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch[{epoch}/{max_epoch}]",
        leave=False,
    )
    for imgs, lbl in bar:
        bar.update()
        imgs, lbl = imgs.cuda(), lbl.cuda()
        logits, _ = model(imgs)
        loss = criterion(logits, lbl)

        optimizer.zero_grad(True)
        # accelerator.backward(loss)
        loss.backward()
        optimizer.step()

        acc = (logits.max(1)[1] == lbl).float().mean()
        bar.set_postfix_str(f"loss:{loss.item():.1f} " f"acc:{acc.item():.1f}")
    bar.close()


def main():
    setup_logger(name="reid_models")
    setup_logger(name="__main__")

    seed = 42
    set_seed(seed)

    accelerator = accelerate.Accelerator(mixed_precision="fp16")

    transform = T.Compose(
        [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor()]
    )
    train_dataset = ImageFolder("datasets/imagenet_train", transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        drop_last=True,
    )

    model_name = "bagtricks_R50_fastreid"
    model = _build_reid_model(model_name, num_classes=1000)
    model = accelerator.prepare(model)

    save_dir = Path(f"logs/imagenet_pretrain")
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    max_epoch = 10
    for epoch in range(1, max_epoch + 1):
        train(
            accelerator,
            model,
            train_loader,
            optimizer,
            criterion,
            max_epoch,
            epoch,
        )

        torch.save(
            model[1].state_dict(),
            save_dir / f"imagenet-{model_name}.pth",
        )


if __name__ == "__main__":
    main()
