"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

import os
from pathlib import Path

import accelerate
import torch
import torch.nn as nn
from prettytable import PrettyTable
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import BatchEasyHardMiner
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from reid_models.data import build_test_dataloaders, build_train_dataloader
from reid_models.evaluate import Estimator
from reid_models.modeling import _build_reid_model
from reid_models.utils import set_seed, setup_logger


def train(
    accelerator,
    model,
    train_loader,
    optimizer,
    miner,
    criterion_t,
    criterion_x,
    max_epoch,
    epoch,
):
    model.train()
    bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch[{epoch}/{max_epoch}]",
        leave=False,
    )
    for batch_idx, (imgs, pids, camids) in enumerate(bar):
        imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()
        logits, feats = model(imgs)

        pairs = miner(feats, pids)
        loss_t = criterion_t(feats, pids, pairs)
        loss_x = criterion_x(logits, pids)

        loss = loss_t + loss_x
        optimizer.zero_grad(True)
        loss.backward()
        # accelerator.backward(loss)
        optimizer.step()

        acc = (logits.max(1)[1] == pids).float().mean()
        bar.set_postfix_str(f"loss:{loss.item():.1f} " f"acc:{acc.item():.1f}")
        bar.update()
    bar.close()


def main():
    setup_logger(name="reid_models")
    logger = setup_logger(name="__main__")

    seed = 42
    set_seed(seed)

    accelerator = accelerate.Accelerator(mixed_precision="fp16")

    test_dataset_names = ["msmt17"]
    test_loaders = build_test_dataloaders(dataset_names=test_dataset_names)
    train_dataset_names = ["msmt17"]
    train_loader = build_train_dataloader(
        dataset_names=train_dataset_names,
        transforms=["randomflip", "randomcrop", "rea"],
        batch_size=64,
        sampler="pk",
        num_instance=4,
        persistent_workers=True,
    )

    model_name = "bagtricks_R50_fastreid"
    num_classes_dict = {"dukemtmcreid": 702, "market1501": 751, "msmt17": 1041}
    num_classes = sum([num_classes_dict[name] for name in train_dataset_names])
    # TODO: Make sure load pretrained model
    os.environ["pretrain"] = "1"
    model = _build_reid_model(
        model_name,
        num_classes=num_classes,
        # weights_path="logs/imagenet_pretrain/imagenet-bagtricks_R50_fastreid.pth",
    )
    model = accelerator.prepare(model)

    save_dir = Path(f"logs/train")
    save_dir.mkdir(parents=True, exist_ok=True)

    max_epoch = 60
    optimizer = torch.optim.Adam(model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=7e-7)

    miner = BatchEasyHardMiner()
    criterion_t = TripletMarginLoss(margin=0.3)
    criterion_x = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, max_epoch + 1):
        train(
            accelerator,
            model,
            train_loader,
            optimizer,
            miner,
            criterion_t,
            criterion_x,
            max_epoch,
            epoch,
        )

        scheduler.step()

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                save_dir / f"{'_'.join(train_dataset_names)}-{model_name}.pth",
            )
            results = PrettyTable(
                field_names=[f"epoch {epoch:0>2}", "top1", "top5", "mAP", "mINP"]
            )
            for name, loader in test_loaders.items():
                cmc, mAP, mINP = Estimator(model, loader[1])(loader[0])
                results.add_row(
                    [
                        name,
                        f"{cmc[0]:.3f}",
                        f"{cmc[4]:.3f}",
                        f"{mAP:.3f}",
                        f"{mINP:.3f}",
                    ]
                )

            logger.info("\n" + str(results))


if __name__ == "__main__":
    main()
