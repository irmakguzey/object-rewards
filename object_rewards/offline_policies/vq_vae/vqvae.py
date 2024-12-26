import einops
import torch
import torch.nn as nn

from third_person_man.models import get_tensor, weights_init_vqvae
from third_person_man.models.vq_vae.residual_vq import ResidualVQ


class VqVae:
    def __init__(
        self,
        input_dim_h=10,  # length of action chunk
        input_dim_w=9,  # action dim
        n_latent_dims=512,
        vqvae_n_embed=32,
        vqvae_groups=4,
        eval=True,
        device="cuda",
        load_dir=None,
        enc_loss_type="skip_vqlayer",
        residual=False,
        encoder_loss_multiplier=1.0,
        act_scale=1.0,
    ):

        self.n_latent_dims = n_latent_dims  # 64
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.rep_dim = self.n_latent_dims
        self.vqvae_n_embed = vqvae_n_embed  # 120
        self.vqvae_lr = 1e-3
        self.vqvae_groups = vqvae_groups
        self.device = device
        self.enc_loss_type = enc_loss_type
        self.residual = residual
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale

        # Initialize the VQ Layer
        discrete_cfg = {"groups": self.vqvae_groups, "n_embed": self.vqvae_n_embed}
        self.vq_layer = ResidualVQ(
            dim=self.n_latent_dims,
            num_quantizers=discrete_cfg["groups"],
            codebook_size=self.vqvae_n_embed,
        ).to(self.device)
        self.embedding_dim = self.n_latent_dims
        self.vq_layer.device = device

        # Initialize the encoder-decoders
        self.encoder = EncoderMLP(
            input_dim=input_dim_w * self.input_dim_h, output_dim=n_latent_dims
        ).to(
            self.device
        )  # NOTE: (irmak) This is the sequential part
        self.decoder = EncoderMLP(
            input_dim=n_latent_dims, output_dim=input_dim_w * self.input_dim_h
        ).to(self.device)

        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.vq_layer.parameters())
        )
        self.vqvae_optimizer = torch.optim.Adam(
            params, lr=self.vqvae_lr, weight_decay=0.0001
        )

        if load_dir is not None:
            try:
                state_dict = torch.load(load_dir)
            except RuntimeError:
                state_dict = torch.load(load_dir, map_location=torch.device("cpu"))
            self.load_state_dict(state_dict)

        if eval:
            self.vq_layer.eval()
        else:
            self.vq_layer.train()

    def draw_logits_forward(self, encoding_logits):
        if self.residual:
            z_embed = self.vq_layer.draw_logits_forward(encoding_logits)
        else:
            z_embed = self.vq_layer.draw_logits_forward(encoding_logits)
        return z_embed

    def draw_code_forward(self, encoding_indices):
        with torch.no_grad():
            if self.residual:
                z_embed = self.vq_layer.get_codes_from_indices(encoding_indices)
                z_embed = z_embed.sum(dim=0)
            else:
                z_embed, encoding_indices = self.vq_layer.draw_code_forward(
                    encoding_indices
                )
        return z_embed

    def get_action_from_latent(self, latent, obs=None):
        output = self.decoder(latent) * self.act_scale
        if self.input_dim_h == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

    def preprocess(self, state):

        if not torch.is_tensor(state):
            state = get_tensor(state, self.device)
        if self.input_dim_h == 1:
            state = state.squeeze(-2)  # state.squeeze(-1)
        else:
            state = einops.rearrange(state, "N T A -> N (T A)")
        return state.to(self.device)

    def get_code(self, state, obs=None, required_recon=False):
        state = state / self.act_scale
        state = self.preprocess(state)
        with torch.no_grad():
            state_rep = self.encoder(state)

            if self.residual:
                state_rep_shape = state_rep.shape[
                    :-1
                ]  # NOTE: I think we're flattening the trajectory here - maybe we don't need a trajectory
                state_rep_flat = state_rep.view(
                    state_rep.size(0), -1, state_rep.size(1)
                )
                state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
                state_vq = state_rep_flat.view(*state_rep_shape, -1)
                vq_code = vq_code.view(*state_rep_shape, -1)
                vq_loss_state = torch.sum(vq_loss_state)
            else:
                state_vq, vq_loss_state, vq_code = self.vq_layer(state_rep)

            if required_recon:
                recon_state = self.decoder(state_vq) * self.act_scale
                recon_state_ae = self.decoder(state_rep) * self.act_scale

                if self.input_dim_h == 1:
                    return state_vq, vq_code, recon_state, recon_state_ae
                else:
                    return (
                        state_vq,
                        vq_code,
                        torch.swapaxes(recon_state, -2, -1),
                        torch.swapaxes(recon_state_ae, -2, -1),
                    )
            else:
                # econ_from_code = self.draw_code_forward(vq_code)
                return state_vq, vq_code

    def vqvae_update(self, state):
        result = self.compute_loss(state)
        rep_loss = result[-1]
        self.vqvae_optimizer.zero_grad()
        rep_loss.backward()
        self.vqvae_optimizer.step()
        return result[:-1]

    def compute_loss(self, state):
        state = state / self.act_scale
        state = self.preprocess(state)
        # print('state after preprocess state.shape: {}'.format(state.shape))
        state_rep = self.encoder(state)
        # print('state_rep.shape: {}'.format(state_rep.shape))

        if self.residual:
            state_rep_shape = state_rep.shape[:-1]
            state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
            # print('state_rep_flat.shape: {}'.format(state_rep_flat.shape))
            state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
            state_vq = state_rep_flat.view(*state_rep_shape, -1)
            vq_code = vq_code.view(*state_rep_shape, -1)
            vq_loss_state = torch.sum(vq_loss_state)
        else:
            state_vq, vq_loss_state, vq_code = self.vq_layer(state_rep)

        dec_out = self.decoder(state_vq)

        encoder_loss = (state - dec_out).abs().mean()
        rep_loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)
        vqvae_recon_loss = torch.nn.MSELoss()(state, dec_out)
        return (
            encoder_loss.clone().detach(),
            vq_loss_state.clone().detach(),
            vq_code,
            vqvae_recon_loss.item(),
            rep_loss,
        )

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.vqvae_optimizer.state_dict(),
            "vq_embedding": self.vq_layer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.vqvae_optimizer.load_state_dict(state_dict["optimizer"])
        self.vq_layer.load_state_dict(state_dict["vq_embedding"])
        self.vq_layer.eval()


class EncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=16,
        hidden_dim=128,
        layer_num=1,
        last_activation=None,
    ):
        super(EncoderMLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layer_num):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        if last_activation is not None:
            self.last_layer = last_activation
        else:
            self.last_layer = None
        self.apply(weights_init_vqvae)

    def forward(self, x):
        h = self.encoder(x)
        state = self.fc(h)
        if self.last_layer:
            state = self.last_layer(state)
        return state


if __name__ == "__main__":

    import torch.utils.data as data
    from einops import rearrange
    from tqdm import tqdm

    from third_person_man.datasets import SeqImageActionDataset

    dataset = SeqImageActionDataset(
        all_data_path="/data/all_dexterity_data/train",
        view_num=0,
        traj_len=10,
        allegro_state_key="hand_tip_states",
    )
    train_loader = data.DataLoader(
        dataset, batch_size=32, shuffle=None, num_workers=4, sampler=None
    )

    vqvae_model = VqVae(
        input_dim_h=10,  # Trajectory length
        input_dim_w=12,  # since it's hand_tip_states
        n_latent_dims=512,
        vqvae_n_embed=16,
        vqvae_groups=2,
        eval=False,
        device="cuda",
        enc_loss_type="through_vqlayer",
        residual=True,
    )

    pbar = tqdm(total=len(train_loader))
    for data in train_loader:
        imgs, act, states = data

        # states = rearrange(states, 'b t d -> b t 1 d') # Maybe try to have the trajectory at first first

        encoder_loss, vq_loss_state, vq_code, vqvae_recon_loss = (
            vqvae_model.vqvae_update(states)
        )  # N T D

        pbar.update(1)
        pbar.set_description(
            "Encoder Loss: {}, VQ Loss State: {}, VQ Code: {}, VQ VAE Recon Loss: {}".format(
                encoder_loss, vq_loss_state, vq_code, vqvae_recon_loss
            )
        )

    pbar.close()
