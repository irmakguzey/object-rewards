import os
from pathlib import Path

import einops
import torch

from .learner import Learner


class VQBET(Learner):
    def __init__(self, bet_wrapper, optimizer):
        self.bet_wrapper = bet_wrapper
        self.optimizer = optimizer

    @property
    def name(self):
        return "vq_bet"

    def to(self, device):
        self.bet_wrapper.to(device)
        self.device = device

    def train(self):
        self.bet_wrapper.train()

    def eval(self):
        self.bet_wrapper.eval()

    def save(self, checkpoint_dir, model_type="best"):
        self.bet_wrapper.save_model(path=Path(checkpoint_dir), model_type=model_type)

    def load(self, checkpoint_dir, training_cfg=None, model_type="best"):
        self.bet_wrapper.load_model(path=Path(checkpoint_dir), model_type=model_type)

    def train_epoch(self, train_loader, epoch, num_train_epochs, logger=None):
        self.train()
        train_loss = 0.0
        epoch_wise_loss_dict = {}
        for batch in train_loader:
            obs, act = [b.to(self.device) for b in batch]
            if epoch < (num_train_epochs * 0.5):
                self.optimizer["optimizer1"].zero_grad()
                self.optimizer["optimizer2"].zero_grad()
            else:
                self.optimizer["optimizer2"].zero_grad()

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
                self.optimizer["optimizer1"].step()
                self.optimizer["optimizer2"].step()
            else:
                self.optimizer["optimizer2"].step()

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
            obs, act = [b.to(self.device) for b in batch]

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
            # logger.log({"eval/epoch_wise_action_diff": action_diff})
            # logger.log({"eval/epoch_wise_action_diff_tot": action_diff_tot})
            # logger.log({"eval/epoch_wise_action_diff_mean_res1": action_diff_mean_res1})
            # logger.log({"eval/epoch_wise_action_diff_mean_res2": action_diff_mean_res2})
            # logger.log({"eval/epoch_wise_action_diff_max": action_diff_max})
            logger.log(
                {"eval/{}".format(x): y for (x, y) in epoch_wise_loss_dict.items()}
            )

        return test_loss / len(test_loader)

    def predict(self, obs, **kwargs):

        with torch.no_grad():
            predicted_act, _, _ = self.bet_wrapper(obs, None, None)
            # print("predicted_act.shape: {}".format(predicted_act.shape))

        # print("predicted_act[0, -1, :].shape: {}".format(predicted_act[-1, 0, :].shape))
        return predicted_act[
            -1, :, :
        ]  # Only get the actions predicted for the last observation
