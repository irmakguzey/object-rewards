import numpy as np
import torch

from object_rewards.utils import OrnsteinUhlenbeckActionNoise, scale_tensor


class OUNoise:
    def __init__(
        self,
        num_expl_steps,
        residual_limit,
    ):
        self.num_expl_steps = num_expl_steps
        self.finger_residual_limit = residual_limit
        self.ou_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(12),  # The mean of the offsets should be 0
            sigma=residual_limit,  # It will give bw -1 and 1 - then this gets multiplied by the scale factors ...
        )

    def explore(
        self, offset_action, global_step, episode_step, device, eval_mode, **kwargs
    ):
        if eval_mode:  # If we are evaluating no exploration
            return offset_action

        print("episode_step: {}".format(episode_step))
        if episode_step == 0:
            print("** RESETTING OU NOISE **")
            self.ou_noise.reset()

        if global_step < self.num_expl_steps:
            print("** EXPLORING **")
            offset_action = torch.FloatTensor(self.ou_noise()).to(device).unsqueeze(0)

            offset_action = scale_tensor(
                tensor=offset_action,
                tensor_min=-self.finger_residual_limit * 2,
                tensor_max=self.finger_residual_limit * 2,
                limit=self.finger_residual_limit,
            )  # Scale the tensor to be around the residual imit
        else:
            print("** ACTING FROM THE MODEL **")

        return offset_action


# This class will explore less and less slowly
# The check for acting from the model should be different and if it's acting from the
# model it will use the model's output as
class ScheduledOUNoise:
    def __init__(
        self,
        num_expl_steps,
        residual_limit,
    ):
        self.num_expl_steps = num_expl_steps
        self.finger_residual_limit = residual_limit
        self.ou_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(12),  # The mean of the offsets should be 0
            sigma=residual_limit,  # It will give bw -1 and 1 - then this gets multiplied by the scale factors ...
        )

        # We will explore once in every step after the global step is higher than expl steps
        self.once_every_explore_step = 1

    def explore(
        self, offset_action, global_step, episode_step, device, eval_mode, **kwargs
    ):
        if eval_mode:  # If we are evaluating no exploration
            return offset_action

        if episode_step == 0:
            print("** RESETTING OU NOISE **")
            self.ou_noise.reset()
            self.arm_ou_noise.reset()

        if global_step < self.num_expl_steps:
            print("** EXPLORING **")

            offset_action = torch.FloatTensor(self.ou_noise()).to(device).unsqueeze(0)

            offset_action = scale_tensor(
                tensor=offset_action,
                tensor_min=-self.finger_residual_limit * 2,
                tensor_max=self.finger_residual_limit * 2,
                limit=self.finger_residual_limit,
            )  # Scale the tensor to be around the residual imit

        else:
            self.set_once_every_explore_step(global_step=global_step)

            if episode_step % self.once_every_explore_step == 0:

                print(f"** EXPLORING ONCE EVERY {self.once_every_explore_step} STEP")

                self.ou_noise.set_previous(
                    offset_action.detach().cpu().numpy().squeeze()
                )

                offset_action = (
                    torch.FloatTensor(self.ou_noise()).to(device).unsqueeze(0)
                )

            else:
                print("** ACTING FROM THE MODEL **")

        return offset_action

    def set_once_every_explore_step(self, global_step):
        if self.num_expl_steps < 1000:
            self.once_every_explore_step = int(
                ((global_step - self.num_expl_steps) / 1000) + 2
            )

        else:
            self.once_every_explore_step = int(
                ((global_step - self.num_expl_steps) / self.num_expl_steps) + 2
            )
