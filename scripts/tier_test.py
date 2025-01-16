import numpy as np



global tiers

tiers = {

            0 : 0.5,

            1 : 0.3,

            2 : 0.125,

            3 : 0.05,

            4 : 0.025

        }

        

def _get_miner_tiers(miner_rewards: list[tuple[int, float]]) -> list[list[int]]:

        if not miner_rewards:

            return []



        _, last_tier_score = miner_rewards[0]



        tiers = [[]]



        for contestant in miner_rewards:

            uid, score = contestant





            if last_tier_score > score * 1.05:

                # New tier

                last_tier_score = score

                tiers.append([])



            tiers[-1].append(uid)



        return list(reversed(tiers))



def _adjust_tiers(num_participants: int) -> None:

        global tiers

        # Sum of original tiers

        total_tiers_sum = sum(tiers.values())

        

        # Get total rewards of the missing tiers

        missing_rewards_sum = sum(tiers[rank] for rank in range(num_participants, len(tiers)))

        

        # Calculate the ratio for adjusting remaining tiers

        adjustment_ratio = (total_tiers_sum - missing_rewards_sum) / total_tiers_sum

        

        # Adjust the remaining tiers proportionally

        adjusted_tiers = {rank: reward * adjustment_ratio for rank, reward in tiers.items() if rank <= num_participants-1}

        

        # Normalize adjusted tiers to ensure they sum up to 1

        adjusted_tiers_sum = sum(adjusted_tiers.values())

        tiers = {rank: reward / adjusted_tiers_sum for rank, reward in adjusted_tiers.items()}



def _get_weights(

        reward_tiers: list[list[int]], 

        uid_submit_date: list[int | None], 

        node_count: int

    ) -> list[float]:

        if not reward_tiers:

            return [1.0] * node_count

        

        def get_submit_date(uid):

            return uid_submit_date.get(uid)



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

            incentive_pool = tiers.get(index, 0)

            

            num_participants = len(tier)

            # Exponential decay applied based on submit date

            decay_weights = np.exp(-np.arange(num_participants))

            normalized_weights = decay_weights / decay_weights.sum()



            # Distribute rewards within the group

            for uid, weight in zip(tier, normalized_weights):

                scores.append((uid, incentive_pool * weight))



        return scores

        

miner_rewards = [[0, 0.8], [1, 0.79], [2, 0.786], [3, 0.781], [4, 0.773], [5, 0.65], [6, 0.64], [7, 0.5]]



ranked_rewards = sorted(miner_rewards, key=lambda x: x[1], reverse=True)

tiered_rewards = _get_miner_tiers(ranked_rewards)



# adjust the weights we assign each tier

num_rewards = len(tiered_rewards) 

if len(tiers) > num_rewards and num_rewards > 0:

    _adjust_tiers(num_rewards)

    

uid_submit_date = {0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7:10}

weights = _get_weights(tiered_rewards, uid_submit_date, len(ranked_rewards))



print(weights)
