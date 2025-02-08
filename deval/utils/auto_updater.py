import time
from threading import Event, Thread
import git
import bittensor as bt
from deval.utils.misc import restart_current_process

UPDATE_RATE_MINUTES = 60


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
                restart_current_process()

    def _check_for_updates(self):
        bt.logging.info("Checking for updates...")
        repo = git.Repo(search_parent_directories=True)
        current_version = repo.head.commit.hexsha

        repo.remotes.origin.pull("main")

        new_version = repo.head.commit.hexsha

        if current_version != new_version:
            bt.logging.info(f"New version detected: '{new_version}'. Restarting...")
            restart_current_process()
        else:
            bt.logging.info("Already up to date.")


    