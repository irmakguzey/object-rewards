import os

import hydra
import torch.optim as optim
from omegaconf import OmegaConf
from pointnet.model import PointNetfeat
from torch.nn.parallel import DistributedDataParallel as DDP

from .vq_bet import VQBET
from .vq_bet_point_net import VQBETPointNet
from .vq_vae import VQVAE


def init_learner(cfg, device, rank=0, **kwargs):

    learner_init_fns = {
        "vq_vae": get_vq_vae_learner,
        "vq_bet": get_vq_bet_learner,
        "vq_bet_point_net": get_vq_bet_point_net_learner,
    }

    learner = learner_init_fns[cfg.learner.type](
        cfg=cfg, device=device, rank=rank, **kwargs
    )

    return learner


def get_vq_vae_learner(cfg, device, rank):

    vqvae_wrapper = hydra.utils.instantiate(cfg.learner.vqvae_wrapper, device=device)

    if cfg.distributed:
        print("VQVAE cannot be trained distributedly so far")
        return None

    learner = VQVAE(vqvae_wrapper=vqvae_wrapper)
    learner.to(device)

    return learner


def get_vq_bet_learner(cfg, device, rank, load=False):

    # Get the vq_vae config
    for dirs in os.walk(cfg.vqvae_load_dir):
        if ".hydra" in dirs[0]:
            vq_vae_cfg_dir = dirs[0]

    vq_vae_cfg = OmegaConf.load(os.path.join(vq_vae_cfg_dir, "config.yaml"))
    cfg.learner.bet_wrapper.vqvae_model.input_dim_h = (
        vq_vae_cfg.learner.vqvae_wrapper.input_dim_h
    )
    cfg.learner.bet_wrapper.vqvae_model.input_dim_w = (
        vq_vae_cfg.learner.vqvae_wrapper.input_dim_w
    )
    cfg.learner.bet_wrapper.vqvae_model.n_latent_dims = (
        vq_vae_cfg.learner.vqvae_wrapper.n_latent_dims
    )
    cfg.learner.bet_wrapper.vqvae_model.vqvae_n_embed = (
        vq_vae_cfg.learner.vqvae_wrapper.vqvae_n_embed
    )
    cfg.learner.bet_wrapper.vqvae_model.vqvae_groups = (
        vq_vae_cfg.learner.vqvae_wrapper.vqvae_groups
    )
    cfg.learner.bet_wrapper.vqvae_model.load_dir = os.path.join(
        cfg.vqvae_load_dir, "models/trained_vqvae_best.pt"
    )

    if load:
        cfg.learner.bet_wrapper.device = device
        cfg.learner.bet_wrapper.vqvae_model.device = device

    bet_wrapper = hydra.utils.instantiate(cfg.learner.bet_wrapper).to(device)

    if cfg.distributed:
        print("VQBET cannot be trained distributedly so far")
        return None

    optimizer = bet_wrapper.configure_optimizers(
        weight_decay=cfg.optimizer.weight_decay,
        learning_rate=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
    )

    learner = VQBET(bet_wrapper=bet_wrapper, optimizer=optimizer)
    learner.to(device)
    return learner


def get_vq_bet_point_net_learner(cfg, device, rank, load=False, **kwargs):
    # Get the vq_vae config
    for dirs in os.walk(cfg.vqvae_load_dir):
        if ".hydra" in dirs[0]:
            vq_vae_cfg_dir = dirs[0]

    vq_vae_cfg = OmegaConf.load(os.path.join(vq_vae_cfg_dir, "config.yaml"))
    cfg.learner.bet_wrapper.vqvae_model.input_dim_h = (
        vq_vae_cfg.learner.vqvae_wrapper.input_dim_h
    )
    cfg.learner.bet_wrapper.vqvae_model.input_dim_w = (
        vq_vae_cfg.learner.vqvae_wrapper.input_dim_w
    )
    cfg.learner.bet_wrapper.vqvae_model.n_latent_dims = (
        vq_vae_cfg.learner.vqvae_wrapper.n_latent_dims
    )
    cfg.learner.bet_wrapper.vqvae_model.vqvae_n_embed = (
        vq_vae_cfg.learner.vqvae_wrapper.vqvae_n_embed
    )
    cfg.learner.bet_wrapper.vqvae_model.vqvae_groups = (
        vq_vae_cfg.learner.vqvae_wrapper.vqvae_groups
    )
    cfg.learner.bet_wrapper.vqvae_model.load_dir = os.path.join(
        cfg.vqvae_load_dir, "models/trained_vqvae_best.pt"
    )

    if load:
        cfg.learner.bet_wrapper.device = device
        cfg.learner.bet_wrapper.vqvae_model.device = device

    bet_wrapper = hydra.utils.instantiate(cfg.learner.bet_wrapper).to(device)

    if cfg.distributed:
        print("VQBET cannot be trained distributedly so far")
        return None

    bet_optimizer = bet_wrapper.configure_optimizers(
        weight_decay=cfg.optimizer.weight_decay,
        learning_rate=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
    )

    point_net = PointNetfeat(global_feat=True)
    point_net_optimizer = optim.Adam(
        point_net.parameters(), lr=0.001, betas=(0.9, 0.999)
    )

    learner = VQBETPointNet(
        point_net=point_net,
        bet_wrapper=bet_wrapper,
        bet_optimizer=bet_optimizer,
        point_net_optimizer=point_net_optimizer,
    )
    learner.to(device)
    return learner
