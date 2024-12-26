from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from third_person_man.models import weight_init
from third_person_man.utils import TruncatedNormal, scale_tensor, schedule

from object_rewards.utils import soft_update_params


class RLLearner(ABC):

    @abstractmethod
    def act(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_actor(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_critic(self, **kwargs):
        raise NotImplementedError

    def update_critic_target(self):
        soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def save_snapshot(self, **kwargs):
        keys_to_save = ["actor", "critic"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v.to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def load_snapshot_eval(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v.to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())


class Identity(nn.Module):
    """
    Author: Janne Spijkervet
    url: https://github.com/Spijkervet/SimCLR
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Actor(nn.Module):
    def __init__(
        self,
        repr_dim,
        action_shape,
        hidden_dim,
        action_repeat,
        normalize_base_actions=False,
        base_action_limits=None,
        device="cuda",
    ):
        self.device = device
        super().__init__()

        # self.base_action_shape = 24 # This is hardcoded
        self.action_repeat = action_repeat

        self.policy = nn.Sequential(
            nn.Linear(repr_dim + action_shape * action_repeat, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape),
        )

        self.apply(weight_init)
        self.normalize_base_actions = normalize_base_actions
        self.base_action_limits = base_action_limits

    def forward(self, obs, action, std):

        # print(f"device: {self.device}")

        action = action.repeat(
            1, self.action_repeat
        )  # Action shape (1, A) -> (1, 100*A)
        if self.normalize_base_actions:
            # print('normalizing base actions - base_action_limits: {}'.format(self.base_action_limits))
            action = scale_tensor(
                action,
                tensor_min=self.base_action_limits[0],
                tensor_max=self.base_action_limits[1],
                limit=1,
            )

        h = torch.cat((obs, action), dim=1)  # h shape: (1, 100*A + Repr_Dim)
        h = h.to(torch.float32)

        mu = self.policy(h)
        mu = torch.tanh(mu)  # NOTE: Remove this?

        std = torch.ones_like(mu) * std

        dist = TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(
        self,
        repr_dim,
        action_shape,
        feature_dim,
        hidden_dim,
        action_repeat,
        is_trunk=False,
        is_residual=False,
        normalize_residual_actions=False,
        residual_limit=None,
        normalize_base_actions=False,
        base_action_limits=None,
    ):
        super().__init__()

        self.is_trunk = is_trunk
        if is_trunk:
            self.trunk = nn.Sequential(
                nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
            )
        else:
            feature_dim = repr_dim

        self.is_residual = is_residual
        if is_residual:
            input_dim = feature_dim + action_shape + action_shape * action_repeat
        else:
            input_dim = feature_dim + action_shape * action_repeat

        self.Q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(
            weight_init
        )  # This function already includes orthogonal weight initialization

        self.action_repeat = action_repeat
        self.normalize_residual_actions = normalize_residual_actions
        self.residual_limit = residual_limit
        self.normalize_base_actions = normalize_base_actions
        self.base_action_limits = base_action_limits

    def forward(self, obs, base_action, residual_action):
        if self.is_trunk:
            h = self.trunk(obs)
        else:
            h = obs

        # Normalize the residual action
        if self.normalize_residual_actions:
            residual_action = scale_tensor(
                residual_action,
                tensor_min=-self.residual_limit,
                tensor_max=self.residual_limit,
                limit=1,
            )
        if self.normalize_base_actions:
            # print('normalizing base actions - base_action_limits: {}'.format(self.base_action_limits))
            base_action = scale_tensor(
                base_action,
                tensor_min=self.base_action_limits[0],
                tensor_max=self.base_action_limits[1],
                limit=1,
            )

        if self.is_residual:
            residual_action = residual_action.repeat(1, self.action_repeat)
            h_action = torch.cat([h, base_action, residual_action], dim=-1)
        else:
            action = base_action + residual_action
            action = action.repeat(1, self.action_repeat)
            h_action = torch.cat([h, action], dim=-1)

        h_action = h_action.to(torch.float32)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2


class DRQv2(RLLearner):
    def __init__(
        self,
        action_shape,
        device,
        actor,
        critic,
        critic_target,
        lr,
        critic_target_tau,
        stddev_schedule,
        stddev_clip,
        residual_limit,
        offset_mask,
        **kwargs,
    ):

        super().__init__()

        # NOTE: For now we don't have scale factors - will need to take a look into that
        # self.action_shape = action_shape
        self.device = device
        self.action_shape = action_shape
        self._set_offset_mask(offset_mask)

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.critic_target = critic_target.to(self.device)
        self.critic_target_tau = critic_target_tau

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.lr = lr
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.stddev_clip = stddev_clip
        self.stddev_schedule = stddev_schedule

        self.residual_limit = residual_limit
        self.lr = lr
        self.train()
        self.critic_target.train()

    def _set_offset_mask(self, offset_mask):
        if self.action_shape == 24:  # just means that the double the fingers are here
            self.offset_mask = torch.concat(
                [torch.Tensor(offset_mask), torch.Tensor(offset_mask)], dim=0
            )
        else:
            self.offset_mask = torch.Tensor(offset_mask)

    def act(self, obs, base_action, global_step, eval_mode, **kwargs):
        stddev = schedule(self.stddev_schedule, global_step)
        dist = self.actor(obs, base_action, stddev)
        offset_action = dist.mean if eval_mode else dist.sample()
        offset_action = scale_tensor(
            tensor=offset_action, tensor_min=-1, tensor_max=1, limit=self.residual_limit
        )
        return offset_action

    def update_critic(
        self,
        obs,
        offset_action,
        base_action,
        base_next_action,
        reward,
        discount,
        next_obs,
        step,
        **kwargs,
    ):

        metrics = dict()

        with torch.no_grad():

            stddev = schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, base_next_action, stddev)
            next_offset_action = dist.sample(clip=self.stddev_clip)
            next_offset_action = scale_tensor(
                tensor=next_offset_action,
                tensor_min=-1,
                tensor_max=1,
                limit=self.residual_limit,
            )
            next_offset_action *= self.offset_mask.to(self.device)

            target_Q1, target_Q2 = self.critic_target(
                next_obs, base_next_action, next_offset_action
            )

            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        offset_action *= self.offset_mask.to(self.device)
        Q1, Q2 = self.critic(obs, base_action, offset_action)

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()

        self.critic_opt.step()

        metrics["train_critic/critic_target_q"] = target_Q.mean().item()
        metrics["train_critic/critic_q1"] = Q1.mean().item()
        metrics["train_critic/critic_q2"] = Q2.mean().item()
        metrics["train_critic/loss"] = critic_loss.item()

        return metrics

    def update_actor(self, obs, base_action, step, **kwargs):
        # print("** UPDATING ACTOR **")

        metrics = dict()

        stddev = schedule(self.stddev_schedule, step)

        dist = self.actor(obs, base_action, stddev)
        offset_action = dist.sample(clip=self.stddev_clip)
        offset_action = scale_tensor(
            tensor=offset_action, tensor_min=-1, tensor_max=1, limit=self.residual_limit
        )
        log_prob = dist.log_prob(offset_action).sum(-1, keepdim=True)

        offset_action *= self.offset_mask.to(self.device)
        Q1, Q2 = self.critic(obs, base_action, offset_action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics["train_actor/loss"] = actor_loss.item()
        metrics["train_actor/actor_logprob"] = log_prob.mean().item()
        metrics["train_actor/actor_ent"] = dist.entropy().sum(dim=-1).mean().item()
        metrics["train_actor/actor_q"] = Q.mean().item()
        metrics["rl_loss"] = -Q.mean().item()

        return metrics

    def save_snapshot(self):
        # TODO: Should I add the optimizers here as well or is it enough to use the model's parameters for the optimizer?
        keys_to_save = ["actor", "critic"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload
