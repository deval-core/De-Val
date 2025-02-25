from huggingface_hub import HfApi, HfFileSystem
from datetime import datetime
import bittensor as bt
import os
from deval.rewards.reward import RewardResult
from deval.task_repository import TASKS
import shutil
from deval.model.chain_metadata import ChainModelMetadataParsed
from substrateinterface import SubstrateInterface
from deval.utils.misc import get_substrate_url
import time

class ModelState:

    def __init__(self, repo_id: str, model_id: str, uid: int, netuid: int):
        self.api = HfApi()
        self.fs = HfFileSystem()

        self.repo_id = repo_id
        self.model_id = model_id 
        self.uid = uid
        self.netuid = netuid
        self.substrate_url = get_substrate_url(netuid)

        
        # defaults
        self.block = None
        self.chain_model_hash = None

        model_url = self.get_model_url()
        for _ in range(3):
            try:
                if len(model_url) >= 3:
                    _ = self.api.model_info(model_url)
                    self.is_valid_repo = True
                else:
                    self.is_valid_repo = False
            except Exception as e:
                time.sleep(10)
                bt.logging.info(f"unable to get model info {e}")
                self.is_valid_repo = False

        if self.is_valid_repo:
            for _ in range(3):
                try:
                    self.last_commit_date: datetime = self.get_last_commit_date()
                    self.last_safetensor_update: datetime = self.get_last_model_update_date()
                except Exception as e:
                    bt.logging.info(f"Unable to get commit or safetensor date {e}")
                    time.sleep(10)

        # reward storage
        self.rewards = {task_name: [] for task_name in TASKS.keys()}


    def _get_safetensor_files(self, model_dir: str | None):
        """Get a sorted list of .safetensors files from the specified directory."""
        if model_dir:
            files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
            return sorted(files)
        else:
            return self.fs.glob(f"{self.get_model_url()}/*.safetensors")

    def add_miner_coldkey(self, coldkey: str):
        self.coldkey = coldkey

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
                bt.logging.info(f"Unable to get last modified date for file {fi}")


        return last_modified_date

    def _get_repo_size(self) -> int:
        files = self.fs.glob(f"{self.get_model_url()}/**")
        sizes = []
        for fi in files:
            file = self.fs.info(fi)
            sz_in_bytes = file['size']
            sz_in_gb = sz_in_bytes * 1e-9
            sizes.append(sz_in_gb)

        return sum(sizes)

    def _get_miner_registration_block(
        self, 
        uid: int,
    ) -> int:
        i = 0 
        backoff = 20

        # add retry logic and sleep 
        while i < 2:

            try:
                substrate = SubstrateInterface(url=self.substrate_url)
                return substrate.query('SubtensorModule', 'BlockAtRegistration', [self.netuid, uid])
            except:
                time.sleep(backoff)
                backoff *= 1.5
            
            i += 1
        
        return substrate.query('SubtensorModule', 'BlockAtRegistration', [self.netuid, uid])


    def should_run_evaluation(
        self, 
        uid: int, 
        max_model_size_gbs: int,
        current_block: int, 
        top_incentive_uids: list[int]) -> bool:
        """
        if the last file submission is after forward start time then we skip. 

        Checks to perform. IF any are true, we return true:
        - check if the miner is in the top incentive UIDs
        - last updated file is from the last 48 hours
        """
        should_evaluate = False

        if uid in top_incentive_uids:
            bt.logging.info(f"In top incentive IDs, continuing with evaluation")
            should_evaluate = True

        if self.is_valid_repo is False:
            bt.logging.info(f"Unable to access repository or Submission was considered invalid - skipping evaluation")
            return False 

        # if the miner was registered 48 hours before the last metadata sync 
        # 14400 blocks per 48 hours 
        n_hours_ago = 420000
        miner_reg_block = self._get_miner_registration_block(uid)
        bt.logging.info(f"block at 48 hours ago: {(current_block - n_hours_ago)} and miner registration block: {miner_reg_block}")
        if  (current_block - n_hours_ago) <= miner_reg_block:
            bt.logging.info("Model registration date within 48 hours, continuing with evaluation")
            should_evaluate = True

        # we can avoid the rest if neither of these are true
        if should_evaluate is not True:
            return False       

        if not self.last_commit_date or not self.last_safetensor_update:
            bt.logging.info(f"Unable to get last commit date: {self.last_commit_date} or last safetensor update: {self.last_safetensor_update}")
            return False
        
        self.repo_size = self._get_repo_size()
        if self.repo_size < 12:
            bt.logging.info(f"Model size is too small - skipping evaluation")
            return False

        if self.repo_size > max_model_size_gbs:
            bt.logging.info(f"Model size is too large - skipping evaluation")
            return False

        return should_evaluate


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

    def add_chain_metadata(self, chain_metadata: ChainModelMetadataParsed | None) -> None:
        if chain_metadata is not None:
            self.block = chain_metadata.block
            self.chain_model_hash = chain_metadata.model_hash

            # the on chain submission must match the miner's model URL
            if chain_metadata.model_url != self.get_model_url():
                bt.logging.info("Chain commit found, but model URL does not match miner")
        else:
            bt.logging.info("No Chain commit found")




    