# %%
from pathlib import Path
import os
import torch
# %%
DATA_PATH=Path("/home/kist/work/workspace/ACM/ASAP/motionData")
FILE_NAME="0-07_02_stageii_run1.pt"

file_path = DATA_PATH / FILE_NAME
data = torch.load(file_path)
# %%

type(data)
data.keys()
data["actions"]
len(data["actions"])
actions = data["actions"]
motion = Path("/home/kist/work/workspace/ACM/motionData/0-07_02_stageii.pkl")
# %%
import joblib
with open(motion, "rb") as f:
    motion_file = joblib.load(f)
# %%
keys = list(motion_file.keys())
motion_data = motion_file[keys[0]]
keys = list(motion_data.keys())
data = motion_data[keys[0]]
# %%
print(len(data))
print(len(actions))
# %%
data_path = "/home/kist/work/workspace/ACM/ASAP/logs/MotionTracking/20260713_kapex_walkmotion_tracking/motions/kapex_walk_20260714_114610.pkl"
with open(data_path, "rb") as f:
    motion_data = joblib.load(f)
motion = motion_data["motion0"]
motion.keys()
len(motion["pose_aa"])
