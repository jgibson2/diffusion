import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from unet import UNet
import utils
from diffusion import DiffusionModel
import torchvision
import torchvision.transforms as TF
import scipy
import matplotlib.pyplot as plt
import PIL.Image as Image


def sample(path):
    time_enc_dim = 6
    max_time_steps = 1000
    image_size = (3, 64, 64)
    batch_size = 32
    beta_schedule = "cosine"

    state = torch.load(path)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    unet = UNet(time_enc_dim=time_enc_dim, predict_scalars=True).to(dev)
    betas = utils.get_named_beta_schedule(beta_schedule, max_time_steps).to(dev)
    model = DiffusionModel(unet, betas, image_size, time_enc_dim=time_enc_dim)
    model.load_state_dict(state["state_dict"])
    model = model.to(dev)

    imgs = model.sample(batch_size, plot=True).detach().cpu()
    grid = torchvision.utils.make_grid(utils.unscale_image_tensor(imgs))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()


def main():
    time_enc_dim = 6
    max_time_steps = 1000
    image_size = (3, 64, 64)
    batch_size = 128
    beta_schedule = "cosine"

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    unet = UNet(time_enc_dim=time_enc_dim, predict_scalars=True).to(dev)
    betas = utils.get_named_beta_schedule(beta_schedule, max_time_steps).to(dev)
    model = DiffusionModel(unet, betas, image_size, time_enc_dim=time_enc_dim)
    model.to(dev)

    transform = TF.Compose((
        utils.ResizedCrop(image_size[1:]),
        TF.RandomHorizontalFlip(p=0.5),
        TF.ToTensor(),
        TF.Lambda(utils.scale_image_tensor),
    ))
    train_dataset = torchvision.datasets.ImageFolder("./data/birds/train", transform=transform)
    val_dataset = torchvision.datasets.ImageFolder("./data/birds/val", transform=transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    # x, _ = next(iter(val_dataloader))
    # imgs = []
    # ts = torch.round(torch.linspace(0, max_time_steps - 1, 10, device=dev)).long()
    # for t in ts:
    #     x_ts, _, _ = model.add_noise(x.to(dev), t * torch.ones((x.shape[0],), dtype=torch.long, device=dev))
    #     x_ts = utils.unscale_scale_image_tensor(x_ts.detach().cpu())
    #     imgs.append(x_ts)
    # grid = torchvision.utils.make_grid(torch.cat(imgs, dim=0), nrow=x.shape[0])
    # plt.imshow(grid.permute(1, 2, 0))
    # plt.show()

    trainer = pl.Trainer(
        gpus=1,
        logger=True,
        callbacks=[pl.callbacks.EarlyStopping(monitor="validation_loss", mode="min", patience=5)],
        enable_checkpointing=True,
        min_epochs=10,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    # main()
    sample("lightning_logs/version_38/checkpoints/epoch=22-step=13362.ckpt")
