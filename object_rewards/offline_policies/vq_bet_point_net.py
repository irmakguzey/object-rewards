import os
from pathlib import Path

import einops
import torch

from .learner import Learner


class VQBETPointNet(Learner):
    def __init__(self, point_net, bet_wrapper, point_net_optimizer, bet_optimizer):
        self.bet_wrapper = bet_wrapper
        self.bet_optimizer = bet_optimizer
        self.point_net_optimizer = point_net_optimizer
        self.point_net = point_net

    @property
    def name(self):
        return "vq_bet_point_net"

    def to(self, device):
        self.bet_wrapper.to(device)
        self.point_net.to(device)
        self.device = device

    def train(self):
        self.bet_wrapper.train()
        self.point_net.train()

    def eval(self):
        self.bet_wrapper.eval()
        self.point_net.eval()

    def save(self, checkpoint_dir, model_type="best"):
        self.bet_wrapper.save_model(path=Path(checkpoint_dir), model_type=model_type)
        torch.save(
            self.point_net.state_dict(),
            os.path.join(checkpoint_dir, f"point_net_{model_type}.pt"),
            _use_new_zipfile_serialization=False,
        )

    def load(self, checkpoint_dir, training_cfg=None, model_type="best"):
        self.bet_wrapper.load_model(path=Path(checkpoint_dir), model_type=model_type)
        point_state_dict = torch.load(
            os.path.join(checkpoint_dir, f"point_net_{model_type}.pt"),
        )
        self.point_net.load_state_dict(point_state_dict)

    def train_epoch(self, train_loader, epoch, num_train_epochs, logger=None):
        self.train()
        train_loss = 0.0
        epoch_wise_loss_dict = {}
        for batch in train_loader:
            pcd, ft_obj, act = [b.to(self.device) for b in batch]
            # print(f'pcd.shape: {pcd.shape}, ft.shape: {ft_obj.shape}, act.shape: {act.shape}')

            traj_len = pcd.shape[1]
            traj_pcd = torch.reshape(pcd, (-1, pcd.shape[-2], pcd.shape[-1]))

            # print(f"traj_pcd.shape: {traj_pcd.shape}")

            # Pass pcd through point net
            pcd_repr = self.point_net(traj_pcd)[0]
            pcd_repr = torch.reshape(pcd_repr, (-1, traj_len, pcd_repr.shape[-1]))
            # print(f"pcd_repr.shape: {pcd_repr.shape}")
            obs = torch.concat([pcd_repr, ft_obj], dim=-1)
            # print(f"in {self.name} learner - obs.shape: {obs.shape}")

            if epoch < (num_train_epochs * 0.5):
                self.bet_optimizer["optimizer1"].zero_grad()
                self.bet_optimizer["optimizer2"].zero_grad()
            else:
                self.bet_optimizer["optimizer2"].zero_grad()
            self.point_net_optimizer.zero_grad()

            predicted_act, loss, loss_dict = self.bet_wrapper(obs, None, act)
            # print("predicted_act.shape: {}".format(predicted_act.shape))
            if not logger is None:
                logger.log(
                    {"train/batch_{}".format(x): y for (x, y) in loss_dict.items()}
                )
                for x, y in loss_dict.items():
                    epoch_wise_key = f"epoch_wise_{x}"
                    if epoch_wise_key in epoch_wise_loss_dict:
                        epoch_wise_loss_dict[epoch_wise_key] += y
                    else:
                        epoch_wise_loss_dict[epoch_wise_key] = y

            # Sum all the losses to return as the total train loss
            train_loss += loss.item()

            # Backward
            loss.backward()
            if epoch < (num_train_epochs * 0.5):
                self.bet_optimizer["optimizer1"].step()
                self.bet_optimizer["optimizer2"].step()
            else:
                self.bet_optimizer["optimizer2"].step()
            self.point_net_optimizer.step()

        if not logger is None:
            logger.log(
                {"train/{}".format(x): y for (x, y) in epoch_wise_loss_dict.items()}
            )

        return train_loss / len(train_loader)

    def test_epoch(self, test_loader, logger=None):
        self.eval()
        test_loss = 0.0
        # action_diff = 0
        # action_diff_tot = 0
        # action_diff_mean_res1 = 0
        # action_diff_mean_res2 = 0
        # action_diff_max = 0
        epoch_wise_loss_dict = {}
        for batch in test_loader:
            pcd, ft_obj, act = [b.to(self.device) for b in batch]
            traj_len = pcd.shape[1]
            traj_pcd = torch.reshape(pcd, (-1, pcd.shape[-2], pcd.shape[-1]))

            # Pass pcd through point net
            pcd_repr = self.point_net(traj_pcd)[0]
            pcd_repr = torch.reshape(pcd_repr, (-1, traj_len, pcd_repr.shape[-1]))
            obs = torch.concat([pcd_repr, ft_obj], dim=-1)

            with torch.no_grad():
                predicted_act, loss, loss_dict = self.bet_wrapper(obs, None, act)

            if not logger is None:
                logger.log(
                    {"eval/batch_{}".format(x): y for (x, y) in loss_dict.items()}
                )
                for x, y in loss_dict.items():
                    epoch_wise_key = f"epoch_wise_{x}"
                    if epoch_wise_key in epoch_wise_loss_dict:
                        epoch_wise_loss_dict[epoch_wise_key] += y
                    else:
                        epoch_wise_loss_dict[epoch_wise_key] = y
                # action_diff += loss_dict["action_diff"]
                # action_diff_tot += loss_dict["action_diff_tot"]
                # action_diff_mean_res1 += loss_dict["action_diff_mean_res1"]
                # action_diff_mean_res2 += loss_dict["action_diff_mean_res2"]
                # action_diff_max += loss_dict["action_diff_max"]

            # Sum all the losses to return as the total train loss
            test_loss += loss

        if not logger is None:
            logger.log(
                {"eval/{}".format(x): y for (x, y) in epoch_wise_loss_dict.items()}
            )

        return test_loss / len(test_loader)

    def predict(self, pcd, ft_obj, **kwargs):

        traj_len = pcd.shape[1]
        traj_pcd = torch.reshape(pcd, (-1, pcd.shape[-2], pcd.shape[-1]))

        with torch.no_grad():
            # Pass pcd through point net
            pcd_repr = self.point_net(traj_pcd)[0]
            pcd_repr = torch.reshape(pcd_repr, (-1, traj_len, pcd_repr.shape[-1]))
            obs = torch.concat([pcd_repr, ft_obj], dim=-1)

            predicted_act, _, _ = self.bet_wrapper(obs, None, None)

        return predicted_act[
            -1, :, :
        ]  # Only get the actions predicted for the last observation
