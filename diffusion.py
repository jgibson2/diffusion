import functools

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import utils
from utils import expand
from matplotlib import pyplot as plt
import tqdm
import einops


class DiffusionModel(pl.LightningModule):
    def __init__(self, model, betas, image_shape, **kwargs):
        '''
        :param model: model to train. Should be NCHW -> NCHW
        :param betas: 1D noise schedule to use. Length determines number of time steps
        :param kwargs: other arguments
        '''
        super(DiffusionModel, self).__init__()
        self.model = model
        self.image_shape = image_shape
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alpha_bars_prev", torch.cat((torch.Tensor([1.0]).to(self.alpha_bars.device),
                                                           self.alpha_bars[:-1]), 0))
        self.max_time_steps = self.betas.shape[0]
        self.ema = utils.ExponentialMovingAverage(self.parameters(), decay=0.99)
        self.time_enc_dim = kwargs.get("time_enc_dim", 5)
        self.dynamic_threshold = kwargs.get("dynamic_threshold", 0.995)

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, batch_idx, "training")

    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, batch_idx, "validation")

    def on_validation_epoch_end(self):
        sampled_imgs = self.sample(32, plot_interval=100)
        grid = torchvision.utils.make_grid(sampled_imgs)
        self.logger.experiment.add_image('generated_images', grid, 0)

    def step(self, batch, batch_idx, phase, image_interval=1000):
        x, _ = batch
        b, c, h, w = x.shape
        ts = torch.randint(0, self.max_time_steps, (b,), device=self.device)
        x_t, epsilon, sigma = self.add_noise(x, ts)
        ts_embedding = utils.timestep_embedding(ts, self.time_enc_dim)
        predicted_epsilon, predicted_v = self.model(x_t.float(), emb=ts_embedding)
        loss = self.compute_losses(x, ts, x_t, epsilon, predicted_epsilon, predicted_v)
        self.log(f"{phase}/loss", loss)
        if batch_idx % image_interval == 0:
            self.logger.experiment.add_image("input_images",
                                             torchvision.utils.make_grid(utils.unscale_image_tensor(x)),
                                             0)
            self.logger.experiment.add_image("sampled_images",
                                             torchvision.utils.make_grid(utils.unscale_image_tensor(x_t)),
                                             0)
            self.logger.experiment.add_image("predicted_noise",
                                             torchvision.utils.make_grid(utils.unscale_image_tensor(predicted_epsilon)),
                                             0)
            self.logger.experiment.add_image("recovered_images",
                                             torchvision.utils.make_grid(
                                                 utils.unscale_image_tensor(
                                                     self.remove_noise(x_t, ts, predicted_epsilon))),
                                             0)
            self.logger.experiment.flush()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model.parameters())

    def add_noise(self, x_0, ts):
        '''
        Add noise to x0, simulating adding noise up to timestep ts
        '''
        epsilon = torch.randn_like(x_0, device=self.device)
        alpha_bar = expand(self.alpha_bars[ts], 4)
        sigma = torch.sqrt(1.0 - alpha_bar)
        x_t = (torch.sqrt(alpha_bar) * x_0) + (sigma * epsilon)
        return x_t, epsilon, sigma

    def remove_noise(self, x_t, ts, epsilon):
        '''
        Remove noise from x_t at timestep ts using epsilon as the noise parameter
        '''
        a = expand(1.0 / torch.sqrt(self.alphas[ts]), 4)
        b = expand(self.betas[ts] / torch.sqrt(1.0 - self.alpha_bars[ts]), 4)
        x_0 = a * (x_t - (b * epsilon))
        return x_0

    def sample(self, n, plot=False, plot_interval=10):
        with torch.inference_mode():
            x = torch.randn((n, *self.image_shape), dtype=torch.float, device=self.device)
            pbar = tqdm.trange(self.max_time_steps - 1, -1, -1)
            for t in pbar:
                ts = t * torch.ones((n,), device=self.device).long()
                ts_embedding = utils.timestep_embedding(ts, self.time_enc_dim)
                pred_eps, pred_vs = self.model(x, emb=ts_embedding)
                mu, sigma = self.sample_posterior_at_timestep(x, ts, pred_eps, pred_vs)
                if t > 0:
                    z = torch.randn_like(x, device=self.device)
                    mu = utils.dynamic_threshold(mu, self.dynamic_threshold).float()
                    # mu = utils.static_threshold(mu)
                    x = mu + (sigma * z)
                else:
                    x = mu
                if self.logger is not None:
                    if t % plot_interval == 0:
                        self.logger.experiment.add_image("generated_progression",
                                                         torchvision.utils.make_grid(
                                                             utils.unscale_image_tensor(x)
                                                         ),
                                                         self.max_time_steps - t)
                if plot:
                    if t % plot_interval == 0:
                        plt.clf()
                        plt.imshow(torchvision.utils.make_grid(
                            utils.unscale_image_tensor(x)
                        ).permute(1, 2, 0).cpu().numpy())
                        plt.axis("off")
                        plt.show(block=False)
                        plt.pause(0.001)
            return x

    def sample_posterior_at_timestep(self, x_t, ts, predicted_epsilons, predicted_vs):
        mu = self.remove_noise(x_t, ts, predicted_epsilons)
        sigma = expand(self.compute_sigmas(ts, predicted_vs.squeeze()), 4)
        return mu, sigma

    def compute_losses(self, x_0, ts, x_t, epsilons, predicted_epsilons, predicted_vs, lm=0.001):
        loss = F.mse_loss(epsilons, predicted_epsilons)
        # loss += lm * self.compute_vlb(xs, ts, x_t, predicted_epsilons, predicted_vs)
        return loss

    def compute_vlb(self, x_0, ts, x_t, predicted_epsilon, predicted_v):
        beta_tilde = expand((1.0 - self.alpha_bars_prev[ts]) / (1.0 - self.alpha_bars[ts]) * self.betas[ts], 2)
        a_tilde = expand(torch.sqrt(self.alpha_bars_prev[ts]) * self.betas[ts] / (1.0 - self.alpha_bars[ts]), 4)
        b_tilde = expand(torch.sqrt(self.alphas[ts]) * (1.0 - self.alpha_bars[ts - 1]) / (1.0 - self.alpha_bars[ts]), 4)
        mu_tilde = (a_tilde * x_0) + (b_tilde * x_t)

        pred_mu = self.remove_noise(x_t, ts, predicted_epsilon).detach()
        pred_sigma = expand(self.compute_sigmas(ts, predicted_v), 2)
        vlb = torch.mean(utils.normal_kl(
            mu_tilde.flatten(start_dim=1),
            torch.log(beta_tilde).flatten(start_dim=1),
            pred_mu.flatten(start_dim=1),
            torch.log(torch.square(pred_sigma)).flatten(start_dim=1)
        ))
        return vlb

    def compute_sigmas(self, ts, predicted_vs):
        # lower bound
        betas = self.betas[ts]
        # upper bound
        beta_tildes = (1.0 - self.alpha_bars_prev[ts]) / (1.0 - self.alpha_bars[ts]) * self.betas[ts]
        # prediction interpolated from lower to upper bound
        sigmas = torch.sqrt(torch.exp((predicted_vs * torch.log(betas)) +
                                      ((1.0 - predicted_vs) * torch.log(beta_tildes))))

        # return torch.sqrt(betas)
        return torch.sqrt(beta_tildes)
        # return sigmas
