from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from beartype.door import die_if_unbearable

sample_number:int = 10
die_if_unbearable(sample_number, int) # should not raise an exception

sample_text: int = "hello" # doesn't raise an exception, but should
try:
    die_if_unbearable(sample_text, int) # will raise an exception since the type is not int
except Exception as e:
    print(e)

dataset_idx:int = 5
# die_if_unbearable(dataset_idx, int)

# repo_id:str = "lerobot/aloha_static_coffee_new" #available_datasets[dataset_idx]
# repo_id:str =  available_datasets[dataset_idx]
repo_id:str = "lerobot/pusht"

print(repo_id)

# die_if_unbearable(repo_id, str)

# We can have a look and fetch its metadata to know more about it:
ds_meta = LeRobotDatasetMetadata(repo_id)

# By instantiating just this class, you can quickly access useful information about the content and the
# structure of the dataset without downloading the actual data yet (only metadata files — which are
# lightweight).
total_episodes: int = ds_meta.total_episodes
print(f"Total number of episodes: {total_episodes}")
avg_frames_per_episode: float = ds_meta.total_frames / total_episodes
print(f"Average number of frames per episode: {avg_frames_per_episode:.3f}")
fps: int = ds_meta.fps
print(f"Frames per second used during data collection: {fps}")
robot_type: str = ds_meta.robot_type
print(f"Robot type: {robot_type}")
camera_keys: list[str] = ds_meta.camera_keys
print(f"keys to access images from cameras: {camera_keys=}\n")

print(ds_meta)

episodes: list[int] = [0]
die_if_unbearable(episodes, list[int])
dataset = LeRobotDataset(repo_id, episodes=episodes)

print(len(dataset))
i=0
print(dataset[i]['task'])

# Push the T-shaped block onto the T-shaped target.