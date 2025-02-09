import torch
import random
import bittensor as bt
from typing import List
from deval.utils.misc import get_substrate_url
from substrateinterface import SubstrateInterface


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


def get_candidate_uids(self, k: int, exclude: List[int] = None) -> list[int]:
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
    hotkeys = getattr(self.metagraph, 'hotkeys')

    for uid in range(self.metagraph.n.item()):
        if uid == self.uid:
            continue

        uid_hotkey = hotkeys[uid]

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
            candidate_uids.append((uid, uid_hotkey))

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    if 0 < len(candidate_uids) < k:
        bt.logging.warning(
            f"Requested {k} uids but only {len(candidate_uids)} were available. To disable this warning reduce the sample size (--neuron.sample_size)"
        )
        return candidate_uids
    elif len(candidate_uids) >= k:
        return random.sample(candidate_uids, k)
    else:
        raise ValueError(f"No eligible uids were found. Cannot return {k} uids")


def fetch_historical_incentive_uids(current_block, lookback = 14400, num_chunks = 25, netuid = 15):
    # defaults allow for 25 chunked interval of incentive from past 48 hours
    substrate = SubstrateInterface(url=get_substrate_url(netuid))
    start_block = current_block - lookback
    batches= [int(start_block + x*(current_block-start_block)/num_chunks) for x in range(num_chunks)]

    uids_with_hist_incentives = set()

    for block_number in batches:
          block_hash = substrate.get_block_hash(block_number)

          if block_hash is None:
              continue  

          incentives = substrate.query(
              'SubtensorModule',
              'Incentive', 
              [15],
              block_hash=block_hash
          )
          uids = [i for i, incentive in enumerate(incentives) if incentive > 0]
          uids_with_hist_incentives.update(uids)
    return list(uids_with_hist_incentives)


def get_top_incentive_uids(
    self, 
    k: int,
    netuid: int,
) -> torch.LongTensor:

    uids = fetch_historical_incentive_uids(self.metagraph.block, netuid=netuid)
    uids = [uid for uid in uids if check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit, [], [])]

    if len(uids) > 0:
        bt.logging.info(f"Top Incentive UIDs: {uids}")
        return torch.tensor(uids)
    else:
        raise ValueError(f"No eligible uids were found. Cannot return {k} uids")