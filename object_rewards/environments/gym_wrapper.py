# Gym implementation of our robotic setup

from collections import deque
from typing import Any, NamedTuple

import dexterous_env
import dm_env
import gymnasium as gym
import numpy as np
from dm_env import StepType, TimeStep, specs
from gymnasium import spaces
from tqdm import tqdm

from object_rewards.utils import turn_homo_to_frames


class RGBArrayAsObservationWrapper(dm_env.Environment):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides

    From: https://github.com/hill-a/stable-baselines/issues/915
    """

    def __init__(self, env, width=84, height=84):
        self._env = env
        self._width = width
        self._height = height
        self._env.reset()

        # print('self._env.render: {}'.format(self._env.render))
        # dummy_obs = self._env.render(mode="rgb_array", width=self._width, height=self._height)
        dummy_obs = self._env.render()
        # print("dummy_obs: {}".format(dummy_obs))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=dummy_obs.dtype
        )
        self.action_space = self._env.action_space

        # Action spec
        wrapped_action_spec = self.action_space
        wrapped_obs_spec = env.observation_space
        if not hasattr(wrapped_action_spec, "minimum"):
            wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
        if not hasattr(wrapped_action_spec, "maximum"):
            wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            np.float32,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )
        # Observation spec
        self._obs_spec = {}

        self._obs_spec["pixels"] = specs.BoundedArray(
            shape=self.observation_space.shape,
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def reset(self, **kwargs):
        obs = {}
        obs = self._env.reset(**kwargs)
        obs["pixels"] = obs["pixels"].astype(np.uint8)
        obs["goal_achieved"] = False
        return obs

    def step(self, action):
        obs, reward, done, _, info = self._env.step(
            action
        )  # I deleted the wrapper truncated that is returning in the time_limit
        obs["pixels"] = obs["pixels"].astype(np.uint8)
        # We will be receiving
        obs["goal_achieved"] = info["is_success"]
        return obs, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        return self._env.render(mode="rgb_array", width=width, height=height)

    def __getattr__(self, name):
        return getattr(self._env, name)


class RobotFeaturesAsObservationWrapper(dm_env.Environment):
    def __init__(self, env, feature_dim=12):
        self._env = env
        self.dim = feature_dim

        # Add the features obs spec
        self._obs_spec = self._env.observation_spec()
        self._obs_spec["features"] = specs.Array(
            shape=(feature_dim,), dtype=np.float32, name="features"
        )

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        obs["features"] = obs["features"].astype(np.float32)
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs["features"] = obs["features"].astype(np.float32)
        return obs, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def render(self, mode="rgb_array", width=256, height=256):
        return self._env.render(mode="rgb_array", width=width, height=height)

    def __getattr__(self, name):
        return getattr(self._env, name)


class RobotConditionAsObservationWrapper(dm_env.Environment):
    def __init__(self, env, condition_dim=2):
        self._env = env
        self.dim = condition_dim

        # self._env.reset()

        # Add the features obs spec
        self._obs_spec = self._env.observation_spec()
        self._obs_spec["condition"] = specs.Array(
            shape=(condition_dim,), dtype=np.float32, name="condition"
        )

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        obs["condition"] = obs["condition"].astype(np.float32)
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs["condition"] = obs["condition"].astype(np.float32)
        return obs, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def render(self, mode="rgb_array", width=256, height=256):
        return self._env.render(mode="rgb_array", width=width, height=height)

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    base_action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getattr__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._features_frames = deque([], maxlen=num_frames)

        wrapped_obs_spec = env.observation_spec()
        self._obs_spec = {}
        if "pixels" in wrapped_obs_spec:
            pixels_shape = wrapped_obs_spec["pixels"].shape
            if len(pixels_shape) == 4:
                pixels_shape = pixels_shape[1:]

            self._obs_spec["pixels"] = specs.BoundedArray(
                shape=np.concatenate(
                    [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
                ),
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name="pixels",
            )

        if "features" in wrapped_obs_spec:
            features_shape = wrapped_obs_spec["features"].shape
            self._obs_spec["features"] = specs.Array(
                shape=(num_frames * features_shape[0],),
                dtype=np.float32,
                name="features",
            )

    def _transform_observation(self, time_step):
        obs = {}
        if "pixels" in self._obs_spec:
            assert len(self._frames) == self._num_frames
            obs["pixels"] = np.concatenate(list(self._frames), axis=0)
        if "features" in self._obs_spec:
            assert len(self._features_frames) == self._num_frames
            obs["features"] = np.concatenate(list(self._features_frames), axis=0)
        obs["goal_achieved"] = time_step.observation["goal_achieved"]
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation["pixels"]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        if "pixels" in self._obs_spec:
            pixels = self._extract_pixels(time_step)
            for _ in range(self._num_frames):
                self._frames.append(pixels)
        if "features" in self._obs_spec:
            features = time_step.observation["features"]
            for _ in range(self._num_frames):
                self._features_frames.append(features)

        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step.observation

        if "pixels" in self._obs_spec:
            pixels = self._extract_pixels(time_step)
            self._frames.append(pixels)
        if "features" in self._obs_spec:
            features = time_step.observation["features"]
            self._features_frames.append(features)

        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0

        # Action spec
        wrapped_action_spec = env.action_space
        if not hasattr(wrapped_action_spec, "minimum"):
            wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
        if not hasattr(wrapped_action_spec, "maximum"):
            wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            np.float32,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        # action = action.astype(self._env.action_space.dtype)
        # Make time step for action space
        observation, reward, done, info = self._env.step(action)
        step_type = StepType.LAST if done else StepType.MID
        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        obs = self._env.reset()
        return TimeStep(  # This only sort of wraps everything in a timestep
            step_type=StepType.FIRST, reward=0, discount=self._discount, observation=obs
        )

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(
        self, action
    ):  # NOTE: Here this is only for returning the base action as a part of the implementation as well
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def _replace(
        self, time_step, observation=None, action=None, reward=None, discount=None
    ):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount

        return ExtendedTimeStep(
            observation=observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=discount,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedH2RTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        # NOTE: Here, you should add a method to return the home state as base action!
        # And that shoud be returned to prevent the jump in the beginning state
        return self._augment_time_step(
            time_step,
            action=np.zeros(
                24,
            ),
            base_action=np.zeros(
                24,
            ),
        )  # NOTE: For now we are hardcoding this - zeros for each axes of the keypoints

    def step(
        self, action, flattened_base_action
    ):  # NOTE: Here this is only for returning the base action as a part of the implementation as well
        time_step = self._env.step(action)
        flattened_action = self._flatten_homo_action(homo_action=action)
        return self._augment_time_step(
            time_step, flattened_action, flattened_base_action
        )

    def _flatten_homo_action(self, homo_action):
        # action: (8,4,4)
        # it will turn that action to (24,) action [will just get the tvecs of the fingers]
        flattened_action = []
        for finger_action in homo_action:
            _, ft_tvec = turn_homo_to_frames(matrix=finger_action)
            flattened_action.append(ft_tvec)

        flattened_action = np.concatenate(flattened_action, axis=0)
        return flattened_action

    def _augment_time_step(self, time_step, action=None, base_action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
            base_action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            base_action=base_action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def _replace(
        self,
        time_step,
        observation=None,
        action=None,
        base_action=None,
        reward=None,
        discount=None,
    ):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if base_action is None:
            base_action = time_step.base_action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount
        return ExtendedTimeStep(
            observation=observation,
            step_type=time_step.step_type,
            action=action,
            base_action=base_action,
            reward=reward,
            discount=discount,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedH2RTimeStepWrapperwArmExploration(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        # NOTE: Here, you should add a method to return the home state as base action!
        # And that shoud be returned to prevent the jump in the beginning state
        return self._augment_time_step(
            time_step,
            action=np.zeros(
                24 + 3,  # 3 is for the arm
            ),
            base_action=np.zeros(
                24 + 3,
            ),
        )  # NOTE: For now we are hardcoding this - zeros for each axes of the keypoints

    def step(
        self, action, flattened_base_action, arm_offset
    ):  # NOTE: Here this is only for returning the base action as a part of the implementation as well
        time_step = self._env.step((action, arm_offset))
        flattened_action = self._flatten_homo_action(homo_action=action)
        flattened_action = np.concatenate([flattened_action, arm_offset], axis=0)
        flattened_base_action = np.concatenate(
            [flattened_base_action, np.zeros(3)], axis=0
        )
        return self._augment_time_step(
            time_step, flattened_action, flattened_base_action
        )

    def _flatten_homo_action(self, homo_action):
        # action: (8,4,4)
        # it will turn that action to (24,) action [will just get the tvecs of the fingers]
        flattened_action = []
        for finger_action in homo_action:
            _, ft_tvec = turn_homo_to_frames(matrix=finger_action)
            flattened_action.append(ft_tvec)

        flattened_action = np.concatenate(flattened_action, axis=0)
        return flattened_action

    def _augment_time_step(self, time_step, action=None, base_action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
            base_action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            base_action=base_action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def _replace(
        self,
        time_step,
        observation=None,
        action=None,
        base_action=None,
        reward=None,
        discount=None,
    ):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if base_action is None:
            base_action = time_step.base_action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount
        return ExtendedTimeStep(
            observation=observation,
            step_type=time_step.step_type,
            action=action,
            base_action=base_action,
            reward=reward,
            discount=discount,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_h2r_envs_custom_params(
    name, img_width, img_height, action_repeat, frame_stack, **kwargs
):

    env = gym.make(name, **kwargs)

    feature_dim = 7 + 16
    env = RGBArrayAsObservationWrapper(env, width=img_width, height=img_height)
    env = RobotFeaturesAsObservationWrapper(
        env, feature_dim=feature_dim
    )  # feature_dim: 16 + 7 for the object
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedH2RTimeStepWrapper(env)
    return env
