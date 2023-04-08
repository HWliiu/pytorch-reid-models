"""
author: Huiwang Liu
e-mail: liuhuiwang1025@outlook.com
"""

from pathlib import Path

import accelerate
import torch
import torch.nn as nn
import torchvision.transforms as T
from kornia.metrics import accuracy
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from reid_models.modeling import _build_reid_model
from reid_models.utils import setup_logger


def train(accelerator, model, train_loader, optimizer, criterion, max_epoch, epoch):
    model.train()
    bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch[{epoch}/{max_epoch}]",
        leave=False,
        disable=not accelerator.is_local_main_process,
    )
    for imgs, lbls in bar:
        bar.update()

        with accelerator.accumulate(model):
            logits, _ = model(imgs)
            loss = criterion(logits, lbls)

            optimizer.zero_grad(True)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(model.parameters(), 1e-1)
            optimizer.step()

        acc = (logits.max(1)[1] == lbls).float().mean()
        bar.set_postfix_str(f"loss:{loss.item():.1f} " f"acc:{acc.item():.1f}")
    bar.close()


def test(accelerator, model, test_loader, epoch):
    # model.eval()
    def set_bn_dropout_eval(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1 or classname.find("Dropout") != -1:
            m.eval()

    model.apply(set_bn_dropout_eval)

    bar = tqdm(
        test_loader,
        total=len(test_loader),
        desc=f"Evaluate Epoch[{epoch}]",
        leave=False,
        disable=not accelerator.is_local_main_process,
    )

    all_logits, all_lbls = [], []
    for imgs, lbls in bar:
        bar.update()

        with torch.no_grad():
            logits, _ = model(imgs)
        logits, lbls = accelerator.gather_for_metrics((logits, lbls))
        all_logits.append(logits)
        all_lbls.append(lbls)

        current_top1, current_top5 = accuracy(logits, lbls, topk=(1, 5))
        bar.set_postfix_str(
            f"top1:{current_top1.item():.1f} top5:{current_top5.item():.1f}"
        )

    all_logits = torch.cat(all_logits)
    all_lbls = torch.cat(all_lbls)
    top1, top5 = accuracy(all_logits, all_lbls, topk=(1, 5))

    bar.close()

    return top1.item(), top5.item()


def main():
    seed = 42
    accelerate.utils.set_seed(seed)

    logs_dir = "logs/imagenet_pretrain"
    kwargs = accelerate.GradScalerKwargs(init_scale=1.0)
    accelerator = accelerate.Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=2,
        project_dir=logs_dir,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[kwargs],
    )

    setup_logger(name="reid_models", distributed_rank=accelerator.local_process_index)
    logger = setup_logger(
        name="__main__", distributed_rank=accelerator.local_process_index
    )

    # build train data
    train_transform = T.Compose(
        [T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor()]
    )
    train_dataset = ImageFolder(
        "datasets/ImageNet2012/train_resized", transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    # build test data
    test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    test_dataset = ImageFolder("datasets/ImageNet2012/val", transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=8,
        pin_memory=True,
    )

    # build model
    model_name = "bagtricks_R50_fastreid"
    model = _build_reid_model(model_name, num_classes=1000)
    # model.load_state_dict(
    #     torch.load(
    #         "logs/imagenet_pretrain/imagenet-bagtricks_R50_fastreid.pth",
    #         map_location="cpu",
    #     )
    # )

    # build optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    # start training
    train_loader, test_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, test_loader, model, optimizer, scheduler
    )

    save_dir = Path(logs_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    max_epoch = 18
    start_epoch = 1
    # accelerator.load_state(accelerator.project_dir)
    for epoch in range(start_epoch, max_epoch + 1):
        # train
        train(
            accelerator,
            model,
            train_loader,
            optimizer,
            criterion,
            max_epoch,
            epoch,
        )
        scheduler.step()

        # save model
        accelerator.save(
            accelerator.get_state_dict(
                accelerate.utils.extract_model_from_parallel(model)
            ),
            save_dir / f"imagenet-{model_name}.pth",
        )
        # accelerator.save_state(accelerator.project_dir)

        # test
        top1, top5 = test(accelerator, model, test_loader, epoch)
        logger.info(
            f"Epoch[{epoch}/{max_epoch}] Results:\ttop1 {top1:.3f}%\ttop5 {top5:.3f}%"
        )


if __name__ == "__main__":
    main()
