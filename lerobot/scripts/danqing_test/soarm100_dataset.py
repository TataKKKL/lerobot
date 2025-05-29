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
repo_id:str = "DanqingZ/so100_test_6"

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

# And see how many frames you have:
print(f"Selected episodes: {dataset.episodes}")
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")

# Or simply load the entire dataset:
full_dataset = LeRobotDataset(repo_id)
print(f"Number of episodes selected: {full_dataset.num_episodes}")
print(f"Number of frames selected: {full_dataset.num_frames}")

# The previous metadata class is contained in the 'meta' attribute of the dataset:
print(full_dataset.meta)

# LeRobotDataset actually wraps an underlying Hugging Face dataset
# (see https://huggingface.co/docs/datasets for more information).
print(full_dataset.hf_dataset)

print(len(dataset))
i=0
print(dataset[i]['task'])


'''
Number of frames selected: 894
LeRobotDatasetMetadata({
    Repository ID: 'DanqingZ/so100_test_6',
    Total episodes: '2',
    Total frames: '894',
    Features: '['action', 'observation.state', 'observation.images.laptop', 'observation.images.phone', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']',
})',

Dataset({
    features: ['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'],
    num_rows: 894
})
447
Grasp a lego block and put it in the bin.
'''