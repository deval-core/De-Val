import torch
import random
import bittensor as bt
from typing import List


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph",
    uid: int,
    vpermit_tao_limit: int,
    coldkeys: set = None,
    ips: set = None,
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
        coldkeys (set): Set of coldkeys to exclude
        ips (set): Set of ips to exclude
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        bt.logging.debug(f"uid: {uid} is not serving")
        return False

    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid] and metagraph.S[uid] > vpermit_tao_limit:
        bt.logging.debug(
            f"uid: {uid} has vpermit and stake ({metagraph.S[uid]}) > {vpermit_tao_limit}"
        )
        return False

    if coldkeys and metagraph.axons[uid].coldkey in coldkeys:
        return False

    if ips and metagraph.axons[uid].ip in ips:
        return False

    # Available otherwise.
    return True


def get_candidate_uids(self, k: int, exclude: List[int] = None) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    coldkeys = set()
    ips = set()
    for uid in range(self.metagraph.n.item()):
        if uid == self.uid:
            continue

        uid_is_available = check_uid_availability(
            self.metagraph,
            uid,
            self.config.neuron.vpermit_tao_limit,
            coldkeys,
            ips,
        )
        if not uid_is_available:
            continue

        if self.config.neuron.query_unique_coldkeys:
            coldkeys.add(self.metagraph.axons[uid].coldkey)

        if self.config.neuron.query_unique_ips:
            ips.add(self.metagraph.axons[uid].ip)

        if exclude is None or uid not in exclude:
            candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    if 0 < len(candidate_uids) < k:
        bt.logging.warning(
            f"Requested {k} uids but only {len(candidate_uids)} were available. To disable this warning reduce the sample size (--neuron.sample_size)"
        )
        return torch.tensor(candidate_uids)
    elif len(candidate_uids) >= k:
        return torch.tensor(random.sample(candidate_uids, k))
    else:
        raise ValueError(f"No eligible uids were found. Cannot return {k} uids")



def get_top_incentive_uids(
    self, 
    k: int,
    num_uids: int = 256
) -> torch.LongTensor:
    # returns the top UIDs within the top_k threshold based on incentive 
    incentive_values = getattr(self.metagraph, 'I')

    incentive_by_uid = [(i,k)  for i, k in zip(range(num_uids), incentive_values)]
    sorted_incentive_by_uid = sorted(incentive_by_uid, key=lambda tup: tup[1], reverse = True)

    # drop to top k and pull out only the uids
    sorted_incentive_by_uid = sorted_incentive_by_uid[:k]
    uids = [uid for uid, _ in sorted_incentive_by_uid]
    uids = [uid for uid in uids if check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit, [], [])]

    if len(uids) > 0:
        return torch.tensor(uids)
    else:
        raise ValueError(f"No eligible uids were found. Cannot return {k} uids")