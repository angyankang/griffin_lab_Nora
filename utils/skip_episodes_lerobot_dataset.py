import json
import pathlib
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class SkipEpisodesLeRobotDataset(Dataset):
    """
    Wrapper for `LeRobotDataset` that skips dirty episodes listed in the `meta/removed_episodes.json` file.
    """
    REMOVED_EPISODES_PATH = "meta/removed_episodes.json"
    REMOVED_EPISODES_KEY = "dirty_episodes"

    def __init__(self, *args, **kwargs):
        self.lerobot_ds = LeRobotDataset(*args, **kwargs)
        
        rm_ep_path = pathlib.Path(self.lerobot_ds.root / self.REMOVED_EPISODES_PATH)
        if rm_ep_path.exists():
            with open(self.lerobot_ds.root / self.REMOVED_EPISODES_PATH) as f:
                self.skip_episodes = json.load(f)[self.REMOVED_EPISODES_KEY]
            self.skip_ranges = [
                range(row["dataset_from_index"], row["dataset_to_index"])
                for row in self.lerobot_ds.meta.episodes if row["episode_index"] in self.skip_episodes
            ]
            # obtain ranges to keep
            self.keep_old_ranges = [
                range(0, self.skip_ranges[0].start),
                *[
                    range(prev_range.stop, curr_range.start)
                    for prev_range, curr_range in zip(self.skip_ranges[:-1], self.skip_ranges[1:])
                    if prev_range.stop < curr_range.start
                ],
                range(self.skip_ranges[-1].stop, len(self.lerobot_ds)),
            ]
        else:
            self.skip_episodes = []
            self.skip_ranges = []
            self.keep_old_ranges = [range(0, len(self.lerobot_ds))]
        # map new indices to old indices
        self.new_ranges_to_old_offsets = {}
        new_range_start = 0
        for old_range in self.keep_old_ranges:
            new_range = range(new_range_start, new_range_start + len(old_range))
            self.new_ranges_to_old_offsets[new_range] = old_range.start - new_range_start
            new_range_start += len(old_range)

    def __len__(self) -> int:
        return list(self.new_ranges_to_old_offsets)[-1].stop

    def __getitem__(self, idx: int) -> dict:
        # find the range that contains the new index
        for new_range, old_offset in self.new_ranges_to_old_offsets.items():
            if idx in new_range:
                return self.lerobot_ds[idx + old_offset]
        raise IndexError(f"Index {idx} out of range")