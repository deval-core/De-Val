import os
import signal
import time
from threading import Event, Thread
import git
import bittensor as bt

UPDATE_RATE_MINUTES = 10


class AutoUpdater:
    _thread: Thread
    _stop_flag: Event

    def __init__(self):
        self._stop_flag = Event()
        self._thread = Thread(target=self._monitor, daemon=True)
        self._check_for_updates()
        self._thread.start()

    def _monitor(self):
        while not self._stop_flag.is_set():
            try:
                current_time = time.localtime()
                if current_time.tm_min % UPDATE_RATE_MINUTES == 0:
                    self._check_for_updates()
                    time.sleep(60)
                else:
                    sleep_minutes = UPDATE_RATE_MINUTES - current_time.tm_min % UPDATE_RATE_MINUTES
                    time.sleep(sleep_minutes * 60 - current_time.tm_sec)
            except Exception as e:
                bt.logging.error(f"Error occurred while checking for updates, attempting to fix the issue by restarting", exc_info=e)
                self._restart()

    def _check_for_updates(self):
        bt.logging.info("Checking for updates...")
        repo = git.Repo(search_parent_directories=True)
        current_version = repo.head.commit.hexsha

        repo.remotes.origin.pull("main")

        new_version = repo.head.commit.hexsha

        if current_version != new_version:
            bt.logging.info(f"New version detected: '{new_version}'. Restarting...")
            self._restart()
        else:
            bt.logging.info("Already up to date.")


    def _restart(self):
        self._stop_flag.set()
        time.sleep(5)
        os.kill(os.getpid(), signal.SIGTERM)

        bt.logging.info("Waiting for process to terminate...")
        for _ in range(60):
            time.sleep(1)
            try:
                os.kill(os.getpid(), 0)
            except ProcessLookupError:
                break

        os.kill(os.getpid(), signal.SIGKILL)