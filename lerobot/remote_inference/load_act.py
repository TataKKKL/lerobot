from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import numpy as np
import torch
import time
from collections import OrderedDict
from typing import Literal
from lerobot.common.policies.act.modeling_act import ACTPolicy
from datetime import datetime


policy = ACTPolicy.from_pretrained("DanqingZ/act_so100_filtered_yellow_cuboid")

print(policy.config.input_features)
num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
num_total_params = sum(p.numel() for p in policy.parameters())
print(f"Number of learnable parameters: {num_learnable_params}")
print(f"Number of total parameters: {num_total_params}")
print(policy.config.image_features)


observation = OrderedDict()
device = "cuda"
state = np.random.randn(6).astype(np.float32)
on_robot_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
phone_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
observation['observation.state'] = torch.from_numpy(state)
observation['observation.images.on_robot'] = torch.from_numpy(on_robot_img)
observation['observation.images.phone'] = torch.from_numpy(phone_img)

for key, value in observation.items():
    # if isinstance(value, np.ndarray):
    # value = torch.from_numpy(value)
    if "image" in key:
        value = value.type(torch.float16) / 255
        value = value.permute(2, 0, 1).contiguous()
    value = value.unsqueeze(0)
    observation[key] = value.to(device)

for key, value in observation.items():
    print(key, value.shape)

for i in range(100):
    start_time = time.time()
    observation["task"] = ["Grasp the yellow cuboid and put it in the bin."]
    action = policy.select_action(observation)
    end_time = time.time()
    duration = end_time - start_time
    duration_ms = duration * 1000
    print(f"Time taken to select action: {duration_ms} ms")
    print(action)