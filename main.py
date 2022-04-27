import click
import matplotlib
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
matplotlib.rcParams['figure.dpi'] = 500
import PIL.Image as Image
import tqdm

time_enc_dim = 6
max_time_steps = 1000
image_size = (3, 96, 64)
batch_size = 48
beta_schedule = "cosine"
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
transform = TF.Compose((
        utils.ResizedCrop(image_size[1:]),
        TF.RandomHorizontalFlip(p=0.5),
        TF.ToTensor(),
        TF.Lambda(utils.scale_image_tensor)
    ))


def sample(path):
    state = torch.load(path)

    unet = UNet(time_enc_dim=time_enc_dim, predict_scalars=True).to(dev)
    betas = utils.get_named_beta_schedule(beta_schedule, max_time_steps).to(dev)
    model = DiffusionModel(unet, betas, image_size, time_enc_dim=time_enc_dim)
    model.load_state_dict(state["state_dict"])
    model = model.to(dev)

    imgs = model.sample(batch_size, plot=True, plot_interval=10).detach().cpu()
    grid = torchvision.utils.make_grid(utils.unscale_image_tensor(imgs))
    plt.clf()
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()


def progression(path):
    state = torch.load(path)

    unet = UNet(time_enc_dim=time_enc_dim, predict_scalars=True).to(dev)
    betas = utils.get_named_beta_schedule(beta_schedule, max_time_steps).to(dev)
    model = DiffusionModel(unet, betas, image_size, time_enc_dim=time_enc_dim)
    model.load_state_dict(state["state_dict"])
    model = model.to(dev)

    train_dataset = torchvision.datasets.ImageFolder("./data/celeba/train", transform=transform)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )

    x, _ = next(iter(train_dataloader))
    b, c, h, w = x.shape
    imgs = []

    steps = max_time_steps

    ts = torch.round(torch.linspace(0, steps, 10, device=dev)).long()
    for t in ts:
        x_ts, _, _ = model.add_noise(x.to(dev), t * torch.ones((x.shape[0],), dtype=torch.long, device=dev))
        x_ts = utils.unscale_image_tensor(x_ts.detach().cpu())
        imgs.append(x_ts)
    with torch.no_grad():
        x = utils.scale_image_tensor(imgs[-1]).float().to(dev)
        for t in tqdm.trange(steps - 1, -1, -1):
            ts = t * torch.ones((b,), device=dev).long()
            ts_embedding = utils.timestep_embedding(ts, time_enc_dim)
            pred_eps, pred_v = model.model(x.float(), emb=ts_embedding)
            mu = model.remove_noise(x, ts, pred_eps)
            z = torch.randn((b, *image_size), device=dev) if t > 0 else torch.zeros(
                (b, *image_size), device=dev)
            sigma = utils.expand(model.compute_sigmas(ts, pred_v), 4)
            x = torch.clamp(mu + (sigma * z), min=-1.0, max=1.0).float()
            if t % (max_time_steps // 10) == 0:
                x_ts = utils.unscale_image_tensor(x.detach().cpu())
                imgs.append(x_ts)
        x_ts = utils.unscale_image_tensor(x.detach().cpu())
        imgs.append(x_ts)

    grid = torchvision.utils.make_grid(torch.cat(imgs, dim=0), nrow=x.shape[0])
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


@click.command()
@click.argument("checkpoint", type=click.Path(), default=None, required=False)
def main(checkpoint):
    if checkpoint is not None:
        sample(checkpoint)
        # progression(checkpoint)
        return

    unet = UNet(time_enc_dim=time_enc_dim, predict_scalars=True).to(dev)
    betas = utils.get_named_beta_schedule(beta_schedule, max_time_steps).to(dev)
    model = DiffusionModel(unet, betas, image_size, time_enc_dim=time_enc_dim)
    model.to(dev)

    train_dataset = torchvision.datasets.ImageFolder("./data/celeba/train", transform=transform)
    val_dataset = torchvision.datasets.ImageFolder("./data/celeba/val", transform=transform)
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

    trainer = pl.Trainer(
        gpus=1,
        logger=True,
        callbacks=[pl.callbacks.EarlyStopping(monitor="validation_loss", mode="min", patience=3)],
        enable_checkpointing=True,
        min_epochs=10,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    main()
