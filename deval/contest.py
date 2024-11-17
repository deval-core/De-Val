from deval.model.model_state import ModelState
from datetime import datetime
from deval.rewards.pipeline import RewardPipeline
import pytz


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

    def validate_model(self, miner_state: ModelState, model_hash: str | None, model_coldkey: str | None) -> bool:
        # ensure the last commit date is before forward start time
        if self.start_time_datetime < miner_state.get_last_commit_date():
            return False

        if not model_hash or not model_coldkey:
            print("Unable to generate model hash or model coldkey, INVALID Model")
            return False

        if model_coldkey != miner_state.coldkey:
            print("Mismatch between the Miner's coldkey and the Model's Coldkey. INVALID Model")
            return False

        # compute the safetensors hash and check if duplicate. Zero out the duplicate based on last safetensors file update
        duplicated_model = self.model_hashes.get(model_hash, None)

        if duplicated_model is None:
            print("No Duplicated model. This is a valid model")
            if miner_state.uid and miner_state.last_safetensor_update:
                self.model_hashes[model_hash] = miner_state
            return True

        else:
            # if duplicates:
            # check which model has the most recent date
            # if current model then that is the duplicate and we return that it is an invalid model
            # if previous model then we zero out the previous model's scores and return this as a valid model
            duplicated_model_uid = duplicated_model.uid
            duplicated_model_date = duplicated_model.last_safetensor_update

            if not duplicated_model_date:
                if miner_state.uid and miner_state.last_safetensor_update:
                    self.model_hashes[model_hash] = miner_state
                print("Found a duplicate model, but unable to find the duplicated date. Weird state. Valid model")
                return True

            if miner_state.last_safetensor_update > duplicated_model_date:
                print("Found a duplicate model and this has a later date. This is an INVALID Model")
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

    def _adjust_tiers(self, num_participants: int) -> None:
        # Sum of original tiers
        total_tiers_sum = sum(self.tiers.values())
        
        # Get total rewards of the missing tiers
        missing_rewards_sum = sum(self.tiers[rank] for rank in range(num_participants, len(self.tiers)))
        
        # Calculate the ratio for adjusting remaining tiers
        adjustment_ratio = (total_tiers_sum - missing_rewards_sum) / total_tiers_sum
        
        # Adjust the remaining tiers proportionally
        adjusted_tiers = {rank: reward * adjustment_ratio for rank, reward in self.tiers.items() if rank <= num_participants}
        
        # Normalize adjusted tiers to ensure they sum up to 1
        adjusted_tiers_sum = sum(adjusted_tiers.values())
        self.tiers = {rank: reward / adjusted_tiers_sum for rank, reward in adjusted_tiers.items()}
        

    def rank_and_select_winners(
        self, 
        task_probabilities: list[tuple()],
    ) -> list[(int, float)]: # (uid, weight)
        """
            takes all the model rewards, ranks them
        """ 
        avg_rewards = []
        denom = sum([num_task for _, num_task in task_probabilities])
        print(f"Denominator: {denom}")
        for uid, scores in self.model_rewards.items():
            print(f"UID: {uid} with scores: {scores}")
            total_scores = [i for values in scores.values() for i in values]
            avg_score = sum(total_scores) / denom
            if avg_score > 0:
                avg_rewards.append((uid, avg_score))

        # rank our rewards and apply weights according to tiers
        ranked_rewards = sorted(avg_rewards, key=lambda x: x[1], reverse=True)

        num_rewards = len(ranked_rewards) 
        if len(self.tiers) > num_rewards and num_rewards > 0:
            self._adjust_tiers(num_rewards)
           
        num_rewards = len(self.tiers)
        weights = [(uid, self.tiers[i]) for i, (uid, _) in enumerate(ranked_rewards[:num_rewards])]

        # spread a small portion of weights across the remaining top incentive miners that had correct answers to maintain ordering
        weights += [(uid, .001) for (uid, _) in ranked_rewards[num_rewards:20]]

        return weights

    def clear_state(self):
        # TODO: 
        # it shouldn't matter, but safe precaution to clear the contest state on next run
        pass