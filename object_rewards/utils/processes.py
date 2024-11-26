import abc
import datetime
import signal
import subprocess
import time

import hydra
from omegaconf import DictConfig


class BaseProcess(abc.ABC):
    def __init__(
        self,
        process_command: list[str],
        reset_duration: datetime.timedelta | None = None,
    ):
        self._reset_duration = reset_duration
        self._process_command = process_command
        self._log_file = None
        self._process = None
        self._is_started = False
        self._started_at = None
        self._reset_count = None

    def reset_if_needed(self):
        if not self._is_started:
            raise ValueError(
                f"Can't reset {self.__class__.__name__}: it's not started yet."
            )
        assert self._reset_count is not None
        self._reset_count += 1
        if self._reset_condition():
            print(f"Resetting {self.__class__.__name__}...")
            self.stop()
            if self._reset_duration is not None:
                print(
                    f"Waiting for {self._reset_duration} to start back {self.__class__.__name__}..."
                )
                time.sleep(self._reset_duration.total_seconds())
            self.start()

    @abc.abstractmethod
    def _reset_condition(self) -> bool:
        pass

    def start(self):
        if self._is_started:
            raise ValueError(
                f"Can't start {self.__class__.__name__}: it's already started."
            )
        print(f"Starting {self.__class__.__name__}...")
        self._log_file = open(f"{self.__class__.__name__}_log.txt", "w")
        self._process = subprocess.Popen(
            self._process_command,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )
        time.sleep(3)
        self._is_started = True
        self._started_at = datetime.datetime.now()
        self._reset_count = 0

    def stop(self):
        if not self._is_started:
            raise ValueError(
                f"Can't stop {self.__class__.__name__}: it' not started yet."
            )
        assert self._process is not None and self._log_file is not None
        print(f"Stopping {self.__class__.__name__}...")
        try:
            self._process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            print(f"{self.__class__.__name__} process already stopped.")
        self._process.wait()
        self._log_file.close()
        print(f"{self.__class__.__name__} stopped.")
        self._is_started = False
        self._started_at = None
        self._reset_count = None


class DeployServer(BaseProcess):
    def __init__(self, script_path: str, reset_every_count: int):
        super().__init__(process_command=["python", script_path], reset_duration=None)
        self._reset_every_count = reset_every_count

    def _reset_condition(self):
        assert self._reset_count is not None
        return self._reset_count >= self._reset_every_count


class AllegroLaunch(BaseProcess):
    def __init__(
        self,
        launch_command: list[str],
        reset_every: datetime.timedelta,
        reset_duration: datetime.timedelta,
    ):
        super().__init__(process_command=launch_command, reset_duration=reset_duration)
        self._reset_every = reset_every

    def _reset_condition(self):
        assert self._started_at is not None
        return datetime.datetime.now() - self._started_at >= self._reset_every


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_online_human_to_robot",
)
def main(cfg: DictConfig) -> None:
    pp = hydra.utils.instantiate(cfg.processes)
    for p in pp:
        p.start()

    for i in range(10):
        input("Press Enter to reset all processes...")
        for p in pp:
            p.reset_if_needed()

    for p in pp:
        p.stop()


if __name__ == "__main__":
    main()
