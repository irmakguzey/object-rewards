import hydra 

from pathlib import Path

from dm_env import specs
from hydra.core.hydra_config import HydraConfig
from PIL import Image as im

from object_rewards.datasets.replay_buffer import *
from object_rewards.utils import *


class Workspace:

    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = Path(cfg.work_dir)

        self._set_up_dist_env()

        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # Initialize hydra
        self.hydra_dir = HydraConfig.get().run.dir

        if self.cfg.buffer_path is None:
            replay_dir = self.work_dir / "buffer" / self.cfg.experiment
        else:
            replay_dir = Path(self.cfg.buffer_path)

        self.buffer_path = (
            replay_dir  # Will be used to calculate the mean / std of the features
        )
        # Initialize the agent
        self._initialize_agent()

        # Start the robot processes
        self._processes = self._setup_processes()

        # Set the environment related parameters
        # The agent will give the initial position of the wrist
        kinova_pose = self.agent.base_policy.initialize_robot_position() # in centimeters
        self._env_setup(kinova_pose)

        # Set the image transform
        self.image_episode_transform = T.Compose(
            [T.ToTensor(), T.Normalize(VISION_IMAGE_MEANS, VISION_IMAGE_STDS)]
        )

        self._global_step = 0
        self._global_episode = 0

        # Set the logger right before the training
        self._set_logger(cfg)

        # Set a timer
        self.timer = FrequencyTimer(cfg.frequency)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def _set_up_dist_env(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"

        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        torch.cuda.set_device(0)

    def _setup_processes(self):
        # return None
        processes = [hydra.utils.instantiate(p_cfg) for p_cfg in self.cfg.processes]
        for p in processes:
            p.start()
        return processes

    def _reset_processes_if_needed(self):
        # return  # NOTE: Change this back
        for p in self._processes:
            p.reset_if_needed()

    def _env_setup(self, kinova_initial_pose):
        self._env_resources = {}

        self.train_env = hydra.utils.call(  # If not call the actual interaction environment
            self.cfg.task.make_fn,
            robot_initial_pose=kinova_initial_pose,
            **self._env_resources,
        )

        # Create replay buffer
        data_specs = [
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array(
                self.train_env.action_spec().shape,
                self.train_env.action_spec().dtype,
                "base_action",
            ),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        ]

        print(f"** BUFFER PATH **: {self.buffer_path}")

        self.replay_storage = ReplayBufferStorage(
            data_specs=data_specs,
            replay_dir=self.buffer_path,  # All the experiments are saved under same name
        )

        self.replay_loader = make_replay_loader_h2r(
            replay_dir=self.buffer_path,
            max_size=self.cfg.replay_buffer_size,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.replay_buffer_num_workers,
            nstep=self.cfg.nstep,
            save_snapshot=self.cfg.save_snapshot,
            discount=self.cfg.discount,
            delta_actions=self.cfg.delta_actions,
            delta_residual_actions=self.cfg.delta_residual_actions,
        )

        self._replay_iter = None

        if self.cfg.evaluate:
            if self.cfg.save_eval_video:
                self.eval_video_recorder = (
                    VideoRecorder(  # It is the same recorder for our case
                        save_dir=Path(self.work_dir)
                        / "online_training_outs/eval_video/videos"
                        / self.cfg.experiment,
                        resize_and_transpose=False,
                    )
                )
        if self.cfg.save_train_video:
            self.train_video_recorder = VideoRecorder(
                save_dir=Path(self.work_dir)
                / "online_training_outs/train_video/videos"
                / self.cfg.experiment,
                resize_and_transpose=False,
            )

    def cleanup(self):
        for p in self._processes:
            p.stop()

    def _initialize_agent(self):

        self.agent = hydra.utils.instantiate(
            self.cfg.agent, buffer_path=self.buffer_path
        )

        print("INITIALIZED AGENT: {}".format(self.agent))

        self.agent.initialize_modules(
            rl_learner_cfg=self.cfg.rl_learner,
            base_policy_cfg=self.cfg.base_policy,
            rewarder_cfg=self.cfg.rewarder,
            explorer_cfg=self.cfg.explorer,
        )

    def _set_logger(self, cfg):
        if self.cfg.log:
            wandb_exp_name = "-".join(self.hydra_dir.split("/")[-2:])
            self.logger = Logger(cfg, wandb_exp_name, out_dir=self.hydra_dir)

    def save_snapshot(self, save_step=False, save_task=False, eval=False):
        snapshot = self.work_dir / "weights"
        snapshot.mkdir(parents=True, exist_ok=True)
        if eval:
            snapshot = snapshot / (
                "snapshot_eval.pt"
                if not save_step
                else f"snapshot_{self.global_step}_eval.pt"
            )
        else:
            snapshot_name = "snapshot.pt"
            if save_task:
                snapshot_name = f"{self.cfg.experiment}_{snapshot_name}"
            if save_step:
                snapshot_name = f"{self.global_step}_{snapshot_name}"
            snapshot = snapshot / snapshot_name

        keys_to_save = ["_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self, snapshot):
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        self.agent.load_snapshot(agent_payload)

    def _add_time_step(self, time_step, time_steps, observations):
        time_steps.append(time_step)

        pil_image_obs = im.fromarray(
            np.transpose(time_step.observation["pixels"], (1, 2, 0)), "RGB"
        )
        transformed_image_obs = self.image_episode_transform(pil_image_obs)
        observations["pil_image_obs"].append(pil_image_obs)
        observations["image_obs"].append(transformed_image_obs)
        observations["features"].append(
            torch.FloatTensor(time_step.observation["features"])
        )

        return time_steps, observations

    def _init_obs(self):
        obs = dict(pil_image_obs=list(), image_obs=list(), features=list())
        return obs

    def _get_act_obs(self, time_step):
        obs_dict = dict(
            image_obs=torch.FloatTensor(time_step.observation["pixels"]) / 255.0,
            features=torch.FloatTensor(time_step.observation["features"]),
        )
        return obs_dict

    def eval(self, evaluation_step):
        step, episode = 0, 0
        eval_until_episode = Until(self.cfg.num_eval_episodes)
        while eval_until_episode(episode):
            episode_step = 0
            is_done = False
            print(f"Eval Episode {episode}")
            time_steps = list()
            observations = self._init_obs()

            print("** RESETTING **")
            self._reset_processes_if_needed()
            time_step = self.train_env.reset()

            time_steps, observations = self._add_time_step(
                time_step, time_steps, observations
            )
            self.eval_video_recorder.init(self._get_image_for_recording(time_step))

            while not (time_step.last() or is_done):
                with torch.no_grad(), eval_mode(self.agent):

                    (
                        action,
                        flattened_base_action,
                        is_done,
                        metrics,
                    ) = self.agent.act(
                        obs=self._get_act_obs(time_step),
                        global_step=self.global_step,
                        episode_step=episode_step,
                        eval_mode=True,
                    )

                time_step = self.train_env.step(
                    action=action, flattened_base_action=flattened_base_action
                )
                time_steps, observations = self._add_time_step(
                    time_step, time_steps, observations
                )
                self.eval_video_recorder.record(
                    self._get_image_for_recording(time_step)
                )
                step += 1
                episode_step += 1

                if self.cfg.log:
                    self.logger.log_metrics(metrics, self.global_step, "global_step")

            episode += 1

            for obs_type in observations.keys():
                if obs_type != "pil_image_obs":
                    observations[obs_type] = torch.stack(observations[obs_type], 0)

            reward = self.agent.get_reward(
                episode_obs=observations,
                episode_id=self.global_episode,
                visualize=self.cfg.save_train_cost_matrices,
            )
            
            rewards_sum = np.sum(reward)
            if self.cfg.log:
                metrics = {"eval_reward": rewards_sum}
                self.logger.log_metrics(
                    metrics,
                    evaluation_step * self.cfg.num_eval_episodes + episode,
                    "eval_step",
                )
            print("EVAL EPISODE: {} - REWARD: {}".format(episode, rewards_sum))

            self.eval_video_recorder.save(
                f"{self.cfg.task.name}_eval_{evaluation_step}_{episode}_r{rewards_sum}.mp4"
            )

            self._reset_processes_if_needed()

    def train_online(self):
        # Set the predicates for training
        train_until_step = Until(self.cfg.num_train_frames)
        seed_until_step = Until(self.cfg.num_seed_frames)
        eval_every_episode = Every(self.cfg.eval_every_episodes)

        episode_step, episode_reward = 0, 0
        observations = self._init_obs()
        time_steps = list()
        print("** RESETTING **")

        time_step = self.train_env.reset()

        self.episode_id = 0
        time_steps, observations = self._add_time_step(
            time_step, time_steps, observations
        )

        self.train_video_recorder.init(self._get_image_for_recording(time_step))
        metrics = dict()
        is_done = False
        next_eval_id = 0
        while train_until_step(self.global_step):

            self.timer.start_loop()

            # At the end of an episode
            if time_step.last() or is_done:

                self._global_episode += 1  # Episode has ended

                # Make each element in observations to torch
                for obs_type in observations.keys():
                    if obs_type != "pil_image_obs":
                        observations[obs_type] = torch.stack(observations[obs_type], 0)

                # Get the reward
                reward = self.agent.get_reward(  # NOTE: There was an error here, fix this
                    episode_obs=observations,
                    episode_id=self.global_episode,
                    visualize=self.cfg.save_train_cost_matrices,
                )

                rewards_sum = np.sum(reward)
                print(
                    "EPISODE: {} - REWARD: {}".format(self._global_episode, rewards_sum)
                )

                # Save the video
                ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                self.train_video_recorder.save(
                    f"{ts}_e{self.global_episode}_s{self.global_step}_r{round(rewards_sum,2)}.mp4"
                )

                # Update the reward
                obs_length = len(time_steps)
                for i, elt in enumerate(time_steps):
                    min_len = (
                        min(obs_length, self.cfg.episode_frame_matches)
                        if self.cfg.episode_frame_matches != -1
                        else len(
                            reward
                        )  # NOTE: There is a problem in here for DTW since that gives more rewards
                    )

                    if (self.cfg.episode_frame_matches == -1
                        or i > (obs_length - min_len)):  # Episode can be shorter than episode_frame_matches
                        new_reward = reward[
                            min_len - (obs_length - i)
                        ]  # Because reward only inclused the matches values so we're doing from the reverse
                        elt = elt._replace(
                            reward=new_reward
                        )  # Update the reward of the object accordingly

                    self.replay_storage.add(elt, last=(i == len(time_steps) - 1))

                # Log
                if self.cfg.log:
                    metrics = {
                        "imitation_reward": rewards_sum,
                        "episode_reward": episode_reward,
                    }
                    self.logger.log_metrics(
                        metrics,
                        time_step=self.global_episode,
                        time_step_name="global_episode",
                    )

                # Reset the environment at the end of the episode
                time_steps = list()
                observations = self._init_obs()

                self._reset_processes_if_needed()
                print("** RESETTING **")

                time_step = self.train_env.reset()

                time_steps, observations = self._add_time_step(
                    time_step, time_steps, observations
                )

                # Checkpoint saving and visualization
                self.train_video_recorder.init(self._get_image_for_recording(time_step))
                if self.cfg.save_snapshot:
                    self.save_snapshot(save_step=False, save_task=True)

                episode_step, episode_reward = 0, 0

            # Eval if needed
            current_eval_id = self.global_episode // self.cfg.eval_every_episodes
            if (
                self.cfg.evaluate
                and eval_every_episode(self.global_episode)
                and current_eval_id >= next_eval_id
            ):
                self.eval(
                    evaluation_step=int(
                        self.global_episode / self.cfg.eval_every_episodes
                    )
                )
                next_eval_id += 1
                print("** RESETTING **")
                self.train_env.reset()

            # Get the action
            with torch.no_grad(), eval_mode(self.agent):

                action, flattened_base_action, is_done, metrics = (
                    self.agent.act(
                        obs=self._get_act_obs(time_step),
                        global_step=self.global_step,
                        episode_step=episode_step,
                        eval_mode=False,
                    )
                )  # Flattened action will be added to the replay buffer inside the gym wrapper

                if self.cfg.log:
                    self.logger.log_metrics(metrics, self.global_step, "global_step")

            print(
                "EPISODE: {} GLOBAL STEP: {} EPISODE STEP: {}".format(
                    self.global_episode, self.global_step, episode_step
                )
            )
            print("---------")

            # Training - update the agents
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(
                    replay_iter=self.replay_iter, step=self.global_step
                )
                if self.cfg.log:
                    self.logger.log_metrics(metrics, self.global_step, "global_step")

            # Take the environment step
            time_step = self.train_env.step(
                action=action, flattened_base_action=flattened_base_action
            )

            episode_reward += time_step.reward
            if self.cfg.log:
                metrics = {"step_reward": time_step.reward}
                self.logger.log_metrics(
                    metrics,
                    time_step=self.global_step,
                    time_step_name="global_step",
                )

            time_steps, observations = self._add_time_step(
                time_step, time_steps, observations
            )

            # Record and increase the steps
            self.train_video_recorder.record(self._get_image_for_recording(time_step))
            episode_step += 1
            self._global_step += 1

            self.timer.end_loop()

    def _get_image_for_recording(self, time_step):
        pixels = time_step.observation["pixels"]
        return pixels


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="train_online_human_to_robot",
)
def main(cfg: DictConfig) -> None:
    workspace = Workspace(cfg)

    if cfg.load_snapshot:
        snapshot = Path(cfg.snapshot_weight)
        print(f"Resuming the snapshot: {snapshot}")
        workspace.load_snapshot(snapshot)
    try:
        workspace.train_online()
    except KeyboardInterrupt:
        print("Stopping online training")
    finally:
        workspace.cleanup()


if __name__ == "__main__":
    main()
