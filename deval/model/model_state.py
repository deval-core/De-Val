from huggingface_hub import HfApi, HfFileSystem
from datetime import datetime, timedelta
import bittensor as bt
import os
from deval.rewards.reward import RewardResult
from deval.task_repository import TASKS
import shutil


class ModelState:

    def __init__(self, repo_id: str, model_id: str, uid: int):
        self.api = HfApi()
        self.fs = HfFileSystem()

        self.repo_id = repo_id
        self.model_id = model_id 
        self.uid = uid

        try:
            _ = self.api.model_info(self.get_model_url())
            self.is_valid_repo = True
        except Exception as e:
            self.is_valid_repo = False

        if self.is_valid_repo:
            self.last_commit_date: datetime | None = self.get_last_commit_date()
            self.last_safetensor_update: datetime | None = self.get_last_model_update_date()

        # reward storage
        self.rewards = {task_name: [] for task_name in TASKS.keys()}
        

    def _get_safetensor_files(self, model_dir: str | None):
        """Get a sorted list of .safetensors files from the specified directory."""
        if model_dir:
            files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
            return sorted(files)
        else:
            return self.fs.glob(f"{self.get_model_url()}/*.safetensors")

    def get_model_url(self):
        return self.repo_id + "/" + self.model_id

    def get_last_commit_date(self) -> datetime | None:
        commits = self.api.list_repo_commits(repo_id=self.get_model_url(), repo_type="model", revision="main")

        if len(commits) > 0:
            return commits[0].created_at
        else: 
            bt.logging.info(f"No commits found for {self.get_model_url()}, return No Commit Date")
            return None

    def get_last_model_update_date(self) -> datetime | None:
        last_modified_date = None

        safetensor_files = self._get_safetensor_files(None)
        for fi in safetensor_files:
            try:
                file = self.fs.ls(fi, detail=True)
                tmp_modified_date = file[0]['last_commit'].date
                if last_modified_date is None:
                    last_modified_date = tmp_modified_date
                elif tmp_modified_date > last_modified_date:
                    last_modified_date = tmp_modified_date
            except:
                bt.logging.info(f"Unable to get last modified date for file {f}")


        return last_modified_date

    def _get_model_size(self) -> int:
        safetensor_files = self._get_safetensor_files(None)
        sizes = []
        for fi in safetensor_files:
            file = self.fs.info(fi)
            sz_in_bytes = file['size']
            sz_in_gb = sz_in_bytes * 1e-9
            sizes.append(sz_in_gb)

        return sum(sizes)



    def should_run_evaluation(
        self, 
        uid: int, 
        max_model_size_gbs: int,
        forward_start_time: int, 
        top_incentive_uids: list[int]) -> bool:
        """
        if the last file submission is after forward start time then we skip. 

        Checks to perform. IF any are true, we return true:
        - check if the miner is in the top incentive UIDs
        - last updated file is from the last 48 hours
        """
        if not self.is_valid_repo:
            bt.logging.info(f"Unable to access repository - skipping evaluation")
            return False

        if self._get_model_size() > max_model_size_gbs:
            bt.logging.info(f"Model size is too large - skipping evaluation")
            return False

        if uid in top_incentive_uids:
            return True

        start_time_datetime = datetime.fromtimestamp(forward_start_time)
        if (start_time_datetime - timedelta(hours=48)) <= self.get_last_commit_date:
            return True
        
        bt.logging.info(f"Did not meet evaluation criteria - skipping evaluation")
        return False


    def cleanup(self, model_dir: str):
        # Remove the model directory to save space
        print(model_dir)
        for filename in os.listdir(model_dir):
            file_path = os.path.join(model_dir, filename)
            print(f"Deleting file: {file_path}")
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def add_reward(self, task_name: str, reward: RewardResult):
        self.rewards[task_name] += reward.rewards



    