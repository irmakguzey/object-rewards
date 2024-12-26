import hydra
import numpy as np
import torch

from object_rewards.utils import (
    flatten_homo_action,
    load_all_episodes,
    merge_all_episodes,
    to_torch,
    turn_frames_to_homo,
    turn_homo_to_frames,
)

torch.set_printoptions(precision=4)


class Agent:

    def __init__(
        self,
        offset_mask,
        features_repeat,
        nstep,
        delta_actions,
        normalize_base_actions,
        normalize_features,
        host,
        experiment_name,
        view_num,
        device,
        update_critic_frequency,
        update_critic_target_frequency,
        update_actor_frequency,
        buffer_path,
        num_expl_steps,
        **kwargs,
    ):

        self.features_repeat = features_repeat
        self.experiment_name = experiment_name
        self.view_num = view_num
        self.device = device
        self.host = host
        self.update_critic_frequency = update_critic_frequency
        self.update_critic_target_frequency = update_critic_target_frequency
        self.update_actor_frequency = update_actor_frequency
        self._set_action_shape_and_offset(offset_mask)
        self.delta_actions = delta_actions
        self.nstep = nstep  # This is used if the delta actions are supposed to be inputted to the model
        self.normalize_base_actions = normalize_base_actions

        self.buffer_path = buffer_path
        self.normalize_features = normalize_features
        self.num_expl_steps = num_expl_steps  # If global step is exceeded the mean and std of the features wil lbe calculated
        self.features_mean, self.features_std = None, None

        self.train()

    @property
    def repr_dim(self):

        # fingertip positions + object mean pose + object trajectory
        repr_dim = (12 + 2 + 3) * self.features_repeat

        return repr_dim

    def _set_action_shape_and_offset(self, offset_mask):
        self.offset_mask = torch.Tensor(offset_mask)
        self.action_shape = self.offset_mask.shape[0]

    def train(self, training=True):
        self.training = training

    def __repr__(self):
        return f"agent"

    def _get_prev_actions(self):
        prev_actions = torch.zeros(self.nstep, self.action_shape).to(self.device)

        base_action, _ = self.base_policy.act(obs=None, episode_step=0)
        base_action = torch.FloatTensor(base_action).to(self.device)

        flattened_base_action = flatten_homo_action(action=base_action).to(self.device)
        shaped_base_action = flattened_base_action[: self.action_shape]

        for nstep in range(self.nstep):
            prev_actions[nstep, :] = shaped_base_action

        return prev_actions

    def _calculate_base_action_limits(self):

        # Will traverse through the demos, get the base actions (should check if it's delta or not)
        # then find the max and min values, and return the max of the absolute of both
        all_actions = []
        i, is_done = 0, False
        if self.delta_actions:
            mock_prev_actions = self._get_prev_actions()

        while not is_done:
            base_action, is_done = self.base_policy.act(obs=None, episode_step=i)
            i += 1
            base_action = torch.FloatTensor(base_action).to(self.device)

            flattened_base_action = flatten_homo_action(action=base_action).to(
                self.device
            )
            shaped_base_action = flattened_base_action[: self.action_shape]
            # Get the delta base action if required
            if self.delta_actions:
                prev_action = mock_prev_actions[0, :]
                model_base_action = shaped_base_action - prev_action

                # Modify the prev actions
                mock_prev_actions = torch.roll(mock_prev_actions, 1, 0)
                mock_prev_actions[-1, :] = shaped_base_action
            else:
                model_base_action = shaped_base_action

            # Get the min max in this base_action
            all_actions.append(model_base_action)

        all_actions = torch.concat(all_actions, dim=0)
        all_actions_min, all_actions_max = torch.min(all_actions), torch.max(
            all_actions
        )

        limit = max(abs(all_actions_min), abs(all_actions_max))

        print(
            "all_actions_min: {}, all_actions_max: {}, limit: {}".format(
                all_actions_min, all_actions_max, limit
            )
        )

        # NOTE: When we returned the actual all actions_min and etc, they tend to
        # be unbalanced and could cause noise
        return -limit, limit

    def _calculate_buffer_mean_and_std(self, key):

        print(f"*** CALCULATING MEAN AND STD OF {key} IN THE BUFFER {self.buffer_path}")

        all_episodes = load_all_episodes(roots=[self.buffer_path])
        all_episode_frames = merge_all_episodes(
            all_episodes=all_episodes, included_keys=[key]
        )

        self.features_mean = torch.FloatTensor(
            np.mean(all_episode_frames[key], axis=0)
        ).to(self.device)
        self.features_std = torch.FloatTensor(
            np.std(all_episode_frames[key], axis=0) + 1.0e-12
        ).to(self.device)

        print(
            f"CALCULATED MEAN: {self.features_mean}, CALCULATED STD: {self.features_std}"
        )

    def initialize_modules(
        self, rl_learner_cfg, base_policy_cfg, rewarder_cfg, explorer_cfg
    ):

        self.base_policy = hydra.utils.instantiate(
            base_policy_cfg,
        )

        self.rewarder = hydra.utils.instantiate(
            rewarder_cfg,
        )

        self.explorer = hydra.utils.instantiate(explorer_cfg)

        rl_learner_cfg.repr_dim = self.repr_dim
        if self.normalize_base_actions:
            action_min, action_max = self._calculate_base_action_limits()
            rl_learner_cfg.base_action_limits = [
                float(action_min),
                float(action_max),
            ]  # This method uses base_policy
        else:
            rl_learner_cfg.base_action_limits = None
        self.rl_learner = hydra.utils.instantiate(rl_learner_cfg)

        # NOTE: We should do this here, because in the init we don't have the base policy
        if self.delta_actions:
            self.prev_actions = self._get_prev_actions()

    def _add_offset_to_action(self, action, offset_action):
        # action: (4,4,4) -> homogenous matrix for each finger + finger orientation
        # offset action: (12,) -> offset in translation for each of the dimensions
        offsetted_action = []
        for i, ft_action in enumerate(action):
            ft_rvec, ft_tvec = turn_homo_to_frames(ft_action)
            finger_id = i % 4
            offset_tvec = offset_action[finger_id * 3 : (finger_id + 1) * 3]

            new_ft_action = turn_frames_to_homo(
                rvec=ft_rvec, tvec=ft_tvec + offset_tvec
            )
            offsetted_action.append(new_ft_action)

        offsetted_action = np.stack(offsetted_action, axis=0)

        # This will be used in the last moment anyways
        return offsetted_action

    def _get_policy_reprs_from_obs(self, features):

        # Normalize the features if desired
        features = features.to(self.device)
        if (not self.features_mean is None) and (not self.features_std is None):
            features = (features - self.features_mean) / self.features_std
        repeated_features = features.repeat(1, self.features_repeat)

        policy_repr = torch.FloatTensor(repeated_features)

        return policy_repr

    def base_act(self, obs, episode_step):
        action, is_done = self.base_policy.act(
            obs=obs, episode_step=episode_step
        )  # action: (8, 4, 4) - is_done: boolean

        return torch.FloatTensor(action).to(self.device), is_done

    def act(self, obs, global_step, episode_step, eval_mode, metrics=None):

        torch.cuda.set_device(self.device)

        with torch.no_grad():
            base_action, is_done = self.base_act(obs, episode_step)

        with torch.no_grad():
            # Get the policy representations
            obs = self._get_policy_reprs_from_obs(
                features=obs["features"].unsqueeze(0),
            )
        flattened_base_action = flatten_homo_action(action=base_action).to(self.device)
        shaped_base_action = flattened_base_action[: self.action_shape]
        # Get the delta base action if required
        if self.delta_actions:
            prev_action = self.prev_actions[0, :]
            model_base_action = shaped_base_action - prev_action

            # Modify the prev actions
            self.prev_actions = torch.roll(self.prev_actions, 1, 0)
            self.prev_actions[-1, :] = shaped_base_action
        else:
            model_base_action = shaped_base_action

        offset_action = self.rl_learner.act(
            obs=obs,
            eval_mode=eval_mode,
            base_action=model_base_action.unsqueeze(0),
            global_step=global_step,
        )

        # Explore the offsets
        offset_action = self.explorer.explore(
            offset_action=offset_action,
            global_step=global_step,
            episode_step=episode_step,
            device=self.device,
            eval_mode=eval_mode,
        )

        # If the exploration is over, calculate the mean and std of the fingertip features
        if (
            self.normalize_features
            and global_step >= self.num_expl_steps
            and self.features_std is None
        ):
            self._calculate_buffer_mean_and_std(key="features")

        # Mask the offset action
        offset_action_np = offset_action.squeeze().detach().cpu()
        offset_action_masked = offset_action_np * self.offset_mask

        # Add a single dimensioned offsets to the homogenous matrix actions
        action_to_apply = self._add_offset_to_action(
            action=base_action.detach().cpu(), offset_action=offset_action_masked
        )

        # If metrics are not None then plot the offsets
        metrics = dict()
        for i in range(len(offset_action_masked)):
            if self.offset_mask[i]:
                if eval_mode:
                    offset_key = f"offsets_eval/offset_{i}"
                else:
                    offset_key = f"offsets_train/offset_{i}"
                metrics[offset_key] = offset_action_masked[i]

        return (
            action_to_apply,
            flattened_base_action.detach().cpu().numpy(),
            is_done,
            metrics,
        )

    def update(self, replay_iter, step):
        metrics = dict()

        torch.cuda.set_device(self.device)

        if (
            step
            % min(
                self.update_critic_frequency,
                self.update_actor_frequency,
                self.update_critic_target_frequency,
            )
            != 0
        ):
            return metrics

        batch = next(replay_iter)
        (
            image_obs,
            features,
            offset_action,
            base_action,
            reward,
            discount,
            next_image_obs,
            next_features,
            base_next_action,
        ) = to_torch(batch, self.device)
        # NOTE: Here all the actions are flattened - that's how things should be in general

        offset_action = offset_action[:, : self.action_shape]
        base_action = base_action[:, : self.action_shape]
        base_next_action = base_next_action[:, : self.action_shape]

        # Get the representations
        # We will return none for representations that are not used in training
        with torch.no_grad():
            obs = self._get_policy_reprs_from_obs(features=features)
            next_obs = self._get_policy_reprs_from_obs(features=next_features)
        metrics["batch_reward"] = reward.mean().item()

        if step % self.update_critic_frequency == 0:
            metrics.update(
                self.rl_learner.update_critic(
                    obs=obs,
                    offset_action=offset_action,
                    base_action=base_action,
                    base_next_action=base_next_action,
                    reward=reward.unsqueeze(1),
                    next_obs=next_obs,
                    discount=discount.unsqueeze(1),
                    step=step,
                )
            )

        if step % self.update_actor_frequency == 0:
            metrics.update(
                self.rl_learner.update_actor(
                    obs=obs, base_action=base_action, step=step
                )
            )

        if step % self.update_critic_target_frequency == 0:
            self.rl_learner.update_critic_target()

        return metrics

    def get_reward(self, episode_obs, episode_id, visualize=False):
        # NOTE: There is no visualization code yet
        reward = self.rewarder.get(
            obs=episode_obs, visualize=visualize, episode_id=episode_id
        )

        if episode_id == 1:
            self.rewarder.update_scale(current_rewards=reward)
            reward = self.rewarder.get(
                obs=episode_obs,
                visualize=False,  # For visualization
                episode_id=episode_id,
            )
        return reward

    def save_snapshot(self):
        return self.rl_learner.save_snapshot()

    def load_snapshot(self, payload):
        return self.rl_learner.load_snapshot(payload)

    def load_snapshot_eval(self, payload):
        return self.rl_learner.load_snapshot_eval(payload)
