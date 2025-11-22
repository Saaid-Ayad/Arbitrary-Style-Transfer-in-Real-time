import utils
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import importlib
import torch
from torch import nn
from itertools import cycle
import matplotlib.pyplot as plt
import shutil
from PIL import Image
import argparse

# %%
import utils
import random
from lightning.pytorch.callbacks import TQDMProgressBar


# %%
from torchvision.transforms import ToTensor, Lambda, RandomCrop
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import ImagesDataset
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train AdaIN Style Transfer Model")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--start_factor",
        type=float,
        default=1.0,
        help="Start factor for LinearLR scheduler",
    )
    parser.add_argument(
        "--end_factor",
        type=float,
        default=0.1,
        help="End factor for LinearLR scheduler",
    )
    parser.add_argument(
        "--content_weight", type=float, default=1.0, help="Content loss weight"
    )
    parser.add_argument(
        "--style_weight", type=float, default=14.0, help="Style loss weight"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to load model weights",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument(
        "--val_interval",
        type=float,
        default=0.25,
        help="Validation interval (fraction of epoch)",
    )
    parser.add_argument(
        "--new_checkpoint_name",
        type=str,
        default="adain_final.pt",
        help="Filename to save the final model checkpoint",
    )
    return parser.parse_args()


transform = [
    ToTensor(),
    Lambda(utils.resizeWithAspectRatio),
    RandomCrop((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
images_train_path = "/kaggle/working/image_train"
images_test_path = "/kaggle/working/test_path"
images_val_path = "/kaggle/working/test_path"
styles_train_path = "/kaggle/working/style_path"
images_train_paths = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(images_train_path)
    for file in files
    if file.endswith(("png", "jpg", "jpeg"))
]
images_val_paths = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(images_val_path)
    for file in files
    if file.endswith(("png", "jpg", "jpeg"))
]
images_test_paths = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(images_test_path)
    for file in files
    if file.endswith(("png", "jpg", "jpeg"))
]
styles_train_paths = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(styles_train_path)
    for file in files
    if file.endswith(("png", "jpg", "jpeg"))
]

train_dataset = ImagesDataset(images_train_paths, transform=transform)
val_dataset = ImagesDataset(images_val_paths, transform=transform)
test_dataset = ImagesDataset(images_test_paths, transform=transform)
styles_dataset = ImagesDataset(
    styles_train_paths, transform=transform
)  # using train images as styles
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Style dataset size: {len(styles_dataset)}")


class AdaINLitModule(L.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        lr,
        scheduler=None,
        scheduler_args=None,
        content_weight=1.0,
        style_weight=1.0,
        checkpoint=None,
    ):  # check if you need to compile
        """
        encoder: nn.Module not compiled
        decoder: nn.Module not compiled
        """
        super().__init__()
        for param in encoder.parameters():
            param.requires_grad = False
        for i, layer in enumerate(encoder):
            if isinstance(layer, nn.ReLU):
                encoder[i] = nn.ReLU(inplace=False)

        # encoder = torch.compile(encoder)
        # decoder = torch.compile(decoder)
        self.encoder = encoder
        self.decoder = decoder
        self.content_weight = content_weight
        self.style_weight = style_weight

        self.model = utils.AdaINModel(encoder, decoder, utils.AdaIN(1e-5))
        # self.model = torch.compile(self.model)

        self.content_loss = utils.ContentLoss(encoder=self.encoder)
        self.style_loss = utils.StyleLoss(encoder=self.encoder)
        # self.content_loss = torch.compile(self.content_loss)
        # self.style_loss = torch.compile(self.style_loss)

        self.lr = lr
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args if scheduler_args is not None else {}
        # self.train_loss = [] # content, style, total
        # self.val_loss = [] # content, style, total

        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint))

    def forward(self, content, style):
        return self.model(content, style)

    def training_step(self, batch, batch_idx):
        content, style = batch
        generated = self.model(content, style)
        y_gen, ada_out = generated["x_gen"], generated["ada_out"]
        c_loss = self.content_loss(y_gen, ada_out)
        s_loss = self.style_loss(y_gen, style)
        loss = c_loss * self.content_weight + s_loss * self.style_weight
        # self.train_loss.append((c_loss.item(),s_loss.item(),loss.item()))
        self.log(f"train_loss", loss, prog_bar=True)
        self.log(f"train_content_loss * {self.content_weight}", c_loss, prog_bar=True)
        self.log(f"train_style_loss * {self.style_weight}", s_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        content, style = batch
        generated = self.model(content, style)
        y_gen, ada_out = generated["x_gen"], generated["ada_out"]
        c_loss = self.content_loss(y_gen, ada_out)
        s_loss = self.style_loss(y_gen, style)
        loss = c_loss * self.content_weight + s_loss * self.style_weight
        # self.val_loss.append((c_loss.item(),s_loss.item(),loss.item()))
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_content_loss", c_loss, prog_bar=True)
        self.log("val_style_loss", s_loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        content, style = batch
        generated = self.model(content, style)
        y_gen, ada_out = generated["x_gen"], generated["ada_out"]
        c_loss = self.content_loss(y_gen, ada_out)
        s_loss = self.style_loss(y_gen, style)
        loss = c_loss * self.content_weight + s_loss * self.style_weight
        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.01
        )
        if self.scheduler is None:
            return optimizer
        return [optimizer], [self.scheduler(optimizer, **self.scheduler_args)]

    def setup(self, stage: str):
        if stage == "fit":
            print(f"Compiling model on rank {self.global_rank}...")
            self.model = torch.compile(self.model)
            self.content_loss = torch.compile(self.content_loss)
            self.style_loss = torch.compile(self.style_loss)

    def train_dataloader(self):
        paths = random.shuffle(images_train_paths)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        styles_loader = DataLoader(
            styles_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        t_loader = zip(train_loader, cycle(styles_loader))
        print(f"Train loader size: {len(train_loader)}")
        return t_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        styles_loader = DataLoader(
            styles_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        v_loader = zip(val_loader, cycle(styles_loader))
        print(f"Val loader size: {len(val_loader)}")
        return v_loader


BATCH_SIZE = None
args = parse_args()
BATCH_SIZE = args.batch_size

# Scheduler
scheduler = torch.optim.lr_scheduler.LinearLR

# Encoder / Decoder
encoder = utils.get_vgg_encoder()
decoder = utils.get_decoder()

# Model
model = AdaINLitModule(
    encoder,
    decoder,
    lr=args.lr,
    scheduler=scheduler,
    scheduler_args={"start_factor": args.start_factor, "end_factor": args.end_factor},
    content_weight=args.content_weight,
    style_weight=args.style_weight,
)

if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))

# Callbacks
bar = TQDMProgressBar(refresh_rate=2)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./checkpoints",
    filename="adain-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
    save_last=True,
    every_n_epochs=1,
)

# Trainer
trainer = L.Trainer(
    devices=args.devices,
    val_check_interval=int(args.val_interval * len(train_dataset) // args.batch_size),
    max_epochs=args.epochs,
    enable_progress_bar=True,
    accumulate_grad_batches=1,
    reload_dataloaders_every_n_epochs=1,
    callbacks=[checkpoint_callback, bar],
)

trainer.fit(model)


# %%
torch.save(model.state_dict(), args.new_checkpoint_name)


# %%
