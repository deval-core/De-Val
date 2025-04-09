from deval.model.model_state import ModelState
from datetime import datetime
from deval.rewards.pipeline import RewardPipeline
import pytz
import numpy as np
from deval.utils.constants import constants


# Note to help with serialization during save, we do not have bittensor package here
# TODO: add a better logger 
class DeValContest:

    def __init__(self, reward_pipeline: RewardPipeline, forward_start_time: int, timeout: int):
        self.model_rewards: dict[int, dict[str, list[float]]] = {} # int = uid, str = task name, list[float] = list of rewards
        self.ranked_rewards: list[tuple(int, float)] = [] # int = uid, float = reward
        self.model_hashes: dict[str, ModelState] = {} 
        self.start_time_datetime: datetime = datetime.fromtimestamp(forward_start_time, tz=pytz.UTC)
        self.reward_pipeline: RewardPipeline = reward_pipeline
        self.timeout: int = timeout

        self.tiers = {
            0 : 0.5,
            1 : 0.3,
            2 : 0.125,
            3 : 0.05,
            4 : 0.025
        }

    def validate_metadata(self, miner_state: ModelState) -> bool:
        # ensure the last commit date is before forward start time
        if self.start_time_datetime < miner_state.get_last_commit_date():
            print(f"Miner's start date {miner_state.get_last_commit_date()} is before validators epoch start time {self.start_time_datetime}")
            return False

        if not miner_state.chain_model_hash or not miner_state.block:
            print(f"Unable to get chain commit data including model hash: {miner_state.chain_model_hash} or block: {miner_state.block}")
            return False

        return True

    def validate_model(
        self, 
        miner_state: ModelState, 
        model_hash: str | None, 
        model_coldkey: str | None, 
        container_size: int,
        max_model_size_in_gbs: int,
    ) -> bool:
        if not self.validate_metadata(miner_state):
            return False

        if not model_hash or not model_coldkey:
            print("Unable to generate model hash or model coldkey, INVALID Model")
            return False

        if model_coldkey != miner_state.coldkey:
            print("Mismatch between the Miner's coldkey and the Model's Coldkey. INVALID Model")
            return False

        if miner_state.chain_model_hash != model_hash:
            print("Mismatch between the model hash on the chain commit and the model hash on huggingface")
            return False

        if container_size > max_model_size_in_gbs:
            print(f"Container too large at {container_size} GBs, failing")
            return False

        # compute the safetensors hash and check if duplicate. Zero out the duplicate based on last safetensors file update
        duplicated_model = self.model_hashes.get(model_hash, None)

        if duplicated_model is None:
            print("No Duplicated model. This is a valid model")
            if miner_state.uid is not None and miner_state.block:
                self.model_hashes[model_hash] = miner_state
            return True

        else:
            # if duplicates:
            # check which model has the most recent date
            # if current model then that is the duplicate and we return that it is an invalid model
            # if previous model then we zero out the previous model's scores and return this as a valid model
            duplicated_model_uid = duplicated_model.uid
            duplicated_model_block = duplicated_model.block

            if not duplicated_model_block:
                if miner_state.uid is not None and miner_state.block:
                    self.model_hashes[model_hash] = miner_state
                print("Found a duplicate model, but unable to find the duplicated date. Weird state. Valid model")
                return True

            if miner_state.block > duplicated_model_block:
                print("Found a duplicate model and this has a commit date later than the duplicate. This is an INVALID Model")
                return False

            else: 
                # if the current model is actually the real one, then we need to update
                # the model that the hash points to and zero out the duplicate rewards 
                # update the model associated 
                self.model_hashes[model_hash] = miner_state
                self.model_rewards.pop(duplicated_model_uid, None)
                print("Found a duplicate model, but this has an earlier commit date and is treated as the valid model")
                return True

                
        
    def update_model_state_with_rewards(self, miner_state: ModelState) -> None:
        self.model_rewards[miner_state.uid] = miner_state.rewards 

    def _get_miner_tiers(self, miner_rewards: list[tuple[int, float]]) -> list[list[int]]:
        if not miner_rewards:
            return []

        _, last_tier_score = miner_rewards[0]

        tiers = [[]]

        for contestant in miner_rewards:
            uid, score = contestant


            if last_tier_score > score * getattr(constants, "tier_improvement_threshold", 1.08):
                # New tier
                last_tier_score = score
                tiers.append([])

            tiers[-1].append(uid)

        return list(reversed(tiers))

    def _adjust_tiers(self, num_participants: int) -> None:
        # Sum of original tiers
        total_tiers_sum = sum(self.tiers.values())
        
        # Get total rewards of the missing tiers
        missing_rewards_sum = sum(self.tiers[rank] for rank in range(num_participants, len(self.tiers)))
        
        # Calculate the ratio for adjusting remaining tiers
        adjustment_ratio = (total_tiers_sum - missing_rewards_sum) / total_tiers_sum
        
        # Adjust the remaining tiers proportionally
        adjusted_tiers = {rank: reward * adjustment_ratio for rank, reward in self.tiers.items() if rank <= num_participants-1}
        
        # Normalize adjusted tiers to ensure they sum up to 1
        adjusted_tiers_sum = sum(adjusted_tiers.values())
        self.tiers = {rank: reward / adjusted_tiers_sum for rank, reward in adjusted_tiers.items()}
        
    def _get_miner_sort_order(self) -> dict[int, int]:
        date_dict = {}
        for _, model_state in self.model_hashes.items():
            uid = model_state.uid
            try:
                date_dict[uid] = model_state.block
            except:
                print("temporary error handling for new validators")
                date_dict[uid] = 10000000000000000000
        return date_dict

    def _get_weights(
        self, 
        reward_tiers: list[list[int]], 
        uid_submit_date: list[int | None], 
        node_count: int
    ) -> list[float]:
        if not reward_tiers:
            return [1.0] * node_count
        
        def get_submit_date(uid):
            return uid_submit_date.get(uid, 10000000000000000000)

        ordered_tiers = [
            sorted(tier, key=get_submit_date) for tier in reward_tiers
        ]

        modified_tiers = []

        last_tier = None

        for tier in reversed(ordered_tiers):
            if last_tier:
                modified_tiers.append([tier[0], *last_tier[1:]])
            else:
                modified_tiers.append([tier[0]])

            last_tier = tier

        if len(last_tier) > 1:
            modified_tiers.append(last_tier[1:])

        # reorder again after modification
        modified_tiers = [
            sorted(tier, key=get_submit_date) for tier in modified_tiers
        ]

        print(f"Final Tiers: {modified_tiers}")
        
        scores = []

        for index, tier in enumerate(modified_tiers):
            incentive_pool = self.tiers.get(index, 0)
            
            num_participants = len(tier)
            # Exponential decay applied based on submit date
            decay_weights = np.exp(-np.arange(num_participants))
            normalized_weights = decay_weights / decay_weights.sum()

            # Distribute rewards within the group
            for uid, weight in zip(tier, normalized_weights):
                scores.append((uid, incentive_pool * weight))

        return scores

    def rank_and_select_winners(
        self, 
        avg_rewards: list[(int, float)],
    ) -> list[(int, float)]: # (uid, weight)
        """
            takes all the model rewards, ranks them
        """ 
        # rank our rewards and apply weights according to tiers
        avg_rewards = [(i, value) for i, value in avg_rewards if value > 0]
        ranked_rewards = sorted(avg_rewards, key=lambda x: x[1], reverse=True)
        print(f"Generated Rewards: {ranked_rewards}")

        # group miners based on min score improvement
        tiered_rewards = self._get_miner_tiers(ranked_rewards)
        print(f"Tiered Rewards: {tiered_rewards}")

        # adjust the weights we assign each tier
        num_rewards = len(tiered_rewards) 
        if len(self.tiers) > num_rewards and num_rewards > 0:
            self._adjust_tiers(num_rewards)
           
        uid_submit_date = self._get_miner_sort_order()
        weights = self._get_weights(tiered_rewards, uid_submit_date, len(ranked_rewards))
        print(f"Computed Weights: {weights}")

        return weights
