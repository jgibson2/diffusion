import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import utils
from matplotlib import pyplot as plt
import tqdm


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
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.max_time_steps = self.betas.shape[0]
        self.ema = utils.ExponentialMovingAverage(self.parameters(), decay=0.9999)
        self.time_enc_dim = kwargs.get("time_enc_dim", 5)

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, batch_idx, "training")

    def validation_step(self, val_batch, batch_idx):
        return self.step(val_batch, batch_idx, "validation")

    def on_validation_epoch_end(self):
        sampled_imgs = self.sample(32)
        grid = torchvision.utils.make_grid(sampled_imgs)
        self.logger.experiment.add_image('generated_images', grid)

    def step(self, batch, batch_idx, phase):
        x, _ = batch
        b, c, h, w = x.shape
        # don't choose 0 to simplify loss calcs (KL-divergence)
        ts = torch.randint(1, self.max_time_steps, (b,), device=self.device)
        x_t, epsilon, sigma = self.add_noise(x, ts)
        ts_embedding = utils.timestep_embedding(ts, self.time_enc_dim)
        predicted_epsilon, predicted_v = self.model(x_t, emb=ts_embedding)
        loss = self.compute_losses(x, ts, x_t, epsilon, predicted_epsilon, predicted_v)
        self.log(f"{phase}_loss", loss)
        if batch_idx % 10 == 0:
            self.logger.experiment.add_image("input_images",
                                             torchvision.utils.make_grid(utils.unscale_image_tensor(x)))
            self.logger.experiment.add_image("sampled_images",
                                             torchvision.utils.make_grid(utils.unscale_image_tensor(x_t)))
            self.logger.experiment.add_image("predicted_noise",
                                             torchvision.utils.make_grid(utils.unscale_image_tensor(predicted_epsilon)))
            self.logger.experiment.add_image("recovered_images",
                                             torchvision.utils.make_grid(
                                                 utils.unscale_image_tensor(
                                                     self.remove_noise(x_t, ts, predicted_epsilon))))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        return optimizer

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model.parameters())

    def add_noise(self, x_0, ts):
        '''
        Add noise to x0, simulating adding noise up to timestep ts
        '''
        b, c, h, w = x_0.shape
        epsilon = torch.randn((b, c, h, w), device=self.device)
        alpha_bar = self.alpha_bars[ts][..., None, None, None]
        sigma = torch.sqrt(1.0 - alpha_bar)
        x_t = (torch.sqrt(alpha_bar) * x_0) + (sigma * epsilon)
        return x_t, epsilon, sigma

    def remove_noise(self, x_t, ts, epsilon):
        '''
        Remove a single step of noise from x_t at timestep ts using epsilon as the noise parameter
        '''
        a = (1.0 / torch.sqrt(self.alphas[ts]))[..., None, None, None]
        b = (self.betas[ts] / torch.sqrt(1.0 - self.alpha_bars[ts]))[..., None, None, None]
        return a * (x_t - (b * epsilon))

    def sample(self, n, plot=False):
        with torch.no_grad():
            x = torch.randn((n, *self.image_shape), device=self.device)
            for t in tqdm.trange(self.max_time_steps - 1, -1, -1):
                ts = t * torch.ones((n,), device=self.device).long()
                ts_embedding = utils.timestep_embedding(ts, self.time_enc_dim)
                pred_eps, pred_v = self.model(x, emb=ts_embedding)
                mu = self.remove_noise(x, ts, pred_eps)

                z = torch.randn((n, *self.image_shape), device=self.device) if t > 0 else torch.zeros(
                    (n, *self.image_shape), device=self.device)
                sigma = self.compute_sigmas(ts, pred_v)[..., None, None, None]
                x = mu + (sigma * z)

                if self.logger is not None:
                    self.logger.experiment.add_image("generated_progression",
                                                     torchvision.utils.make_grid(utils.unscale_image_tensor(mu)),
                                                     self.max_time_steps - t)
                if plot:
                    if t % 10 == 0:
                        plt.imshow(torchvision.utils.make_grid(
                            utils.unscale_image_tensor(mu)
                        ).permute(1, 2, 0).cpu().numpy())
                        plt.show(block=False)
                        plt.pause(0.001)

            return torch.clamp(x, min=-1.0, max=1.0)

    def compute_losses(self, xs, ts, x_t, epsilons, predicted_epsilons, predicted_vs, lm=0.001):
        loss = F.mse_loss(epsilons, predicted_epsilons)
        # loss += lm * self.compute_vlb(xs, ts, x_t, predicted_epsilons, predicted_vs)
        return loss

    def compute_vlb(self, xs, ts, x_t, predicted_epsilon, predicted_v):
        beta_tilde = ((1.0 - self.alpha_bars[ts - 1]) / (1.0 - self.alpha_bars[ts]) * self.betas[ts])[..., None]
        a_tilde = (torch.sqrt(self.alpha_bars[ts - 1]) * self.betas[ts] / (1.0 - self.alpha_bars[ts]))[..., None, None, None]
        b_tilde = (torch.sqrt(self.alphas[ts]) * (1.0 - self.alpha_bars[ts - 1]) / (1.0 - self.alpha_bars[ts]))[..., None, None, None]
        mu_tilde = (a_tilde * xs) + (b_tilde * x_t)

        pred_mu = self.remove_noise(x_t, ts, predicted_epsilon).detach()
        pred_sigma = self.compute_sigmas(ts, predicted_v)[..., None]
        vlb = torch.mean(utils.normal_kl(
            mu_tilde.flatten(start_dim=1),
            torch.log(beta_tilde).flatten(start_dim=1),
            pred_mu.flatten(start_dim=1),
            torch.log(torch.square(pred_sigma)).flatten(start_dim=1)
        ))
        return vlb

    def compute_sigmas(self, ts, predicted_vs):
        betas = self.betas[ts]
        return torch.sqrt(betas)

        # beta_tildes = ((1.0 - self.alpha_bars[ts - 1]) / (1.0 - self.alpha_bars[ts]) * self.betas[ts])
        # predicted_sigmas = torch.sqrt(torch.exp((predicted_vs * torch.log(betas)) +
        #                              ((1.0 - predicted_vs) * torch.log(beta_tildes))))
        # return predicted_sigmas
