from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import numpy as np
import torch
import time
from collections import OrderedDict
from typing import Literal
from lerobot_client import LeRobotClient
from lerobot.common.policies.act.modeling_act import ACTPolicy
from datetime import datetime

# https://www.physicalintelligence.company/blog/openpi, 2/4
# https://huggingface.co/blog/pi0

policy = PI0Policy.from_pretrained("DanqingZ/pi0_so100_test_yellow_cuboid_2_20250603_102352")
# policy.to("cuda")
# policy.eval()
print(policy.config.input_features)
num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
num_total_params = sum(p.numel() for p in policy.parameters())
print(f"Number of learnable parameters: {num_learnable_params}")
print(f"Number of total parameters: {num_total_params}")
print(policy.config.image_features)

'''
{'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>, shape=(6,)), 'observation.images.on_robot': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640)), 'observation.images.phone': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640))}
Number of learnable parameters: 3088929824
Number of total parameters: 3501372212
{'observation.images.on_robot': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640)), 'observation.images.phone': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640))}
'''

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
observation["task"] = ["Grasp the yellow cuboid and put it in the bin."]
action = policy.select_action(observation)
print(action)

'''
{'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>, shape=(6,)), 'observation.images.on_robot': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640)), 'observation.images.phone': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640))}
Number of learnable parameters: 3088929824
Number of total parameters: 3501372212
{'observation.images.on_robot': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640)), 'observation.images.phone': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640))}
observation.state torch.Size([1, 6])
observation.images.on_robot torch.Size([1, 3, 480, 640])
observation.images.phone torch.Size([1, 3, 480, 640])
tensor([[ -4.5581,  67.8996,  75.5150,  23.5823,   6.5133, -10.0830]],
       device='cuda:0')
'''


