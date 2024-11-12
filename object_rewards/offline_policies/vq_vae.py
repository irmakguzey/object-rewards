import os

import torch

from .learner import Learner


class VQVAE(Learner):
    def __init__(self, vqvae_wrapper):
        self.vqvae_wrapper = vqvae_wrapper

    @property
    def name(self):
        return "vq_vae"

    def to(self, device):
        self.device = device
        self.vqvae_wrapper.vq_layer = self.vqvae_wrapper.vq_layer.to(device)
        self.vqvae_wrapper.encoder = self.vqvae_wrapper.encoder.to(device)
        self.vqvae_wrapper.decoder = self.vqvae_wrapper.decoder.to(device)

    def train(self):
        self.vqvae_wrapper.vq_layer.train()
        self.vqvae_wrapper.encoder.train()
        self.vqvae_wrapper.decoder.train()

    def eval(self):
        self.vqvae_wrapper.vq_layer.eval()
        self.vqvae_wrapper.encoder.eval()
        self.vqvae_wrapper.decoder.eval()

    def save(self, checkpoint_dir, model_type="best"):
        state_dict = self.vqvae_wrapper.state_dict()
        torch.save(
            state_dict,
            os.path.join(checkpoint_dir, f"trained_vqvae_{model_type}.pt"),
            _use_new_zipfile_serialization=False,
        )

    def load(self, checkpoint_dir, training_cfg=None, model_type="best"):
        # Get the weights
        state_dict = torch.load(
            os.path.join(checkpoint_dir, f"trained_vqvae_{model_type}.pt")
        )
        # NOTE: This wouldn't work with multiple gpus just fyi
        self.vqvae_wrapper.load_state_dict(state_dict)

    def train_epoch(self, train_loader, logger=None, **kwargs):
        self.train()
        train_loss = 0.0
        for batch in train_loader:
            # Get the trajectory states
            _, actions = [b.to(self.device) for b in batch]
            # Update the VQ_VAE
            encoder_loss, vq_loss_state, _, vqvae_recon_loss = (
                self.vqvae_wrapper.vqvae_update(actions)
            )

            if not logger is None:
                logger.log({"train/batch_encoder_loss": encoder_loss})
                logger.log({"train/batch_vq_loss_state": vq_loss_state})
                logger.log({"train/batch_vqvae_recon_loss": vqvae_recon_loss})

            # Sum all the losses to return as the total train loss
            # NOTE: Make sure that these losses are actually meant to be summed like this
            train_loss += encoder_loss + vq_loss_state + vqvae_recon_loss

        return train_loss / len(train_loader)

    def test_epoch(self, test_loader, logger=None):
        self.eval()
        test_loss = 0.0
        for batch in test_loader:
            # Get the trajectory states
            _, actions = [b.to(self.device) for b in batch]

            with torch.no_grad():
                # Update the VQ_VAE
                encoder_loss, vq_loss_state, _, vqvae_recon_loss, _ = (
                    self.vqvae_wrapper.compute_loss(actions)
                )

            if not logger is None:
                # logger.log({"train/{}".format(x): y for (x, y) in loss_dict.items()})
                logger.log({"eval/batch_encoder_loss": encoder_loss})
                logger.log({"eval/batch_vq_loss_state": vq_loss_state})
                logger.log({"eval/batch_vqvae_recon_loss": vqvae_recon_loss})

            test_loss += encoder_loss + vq_loss_state + vqvae_recon_loss

        return test_loss / len(test_loader)
