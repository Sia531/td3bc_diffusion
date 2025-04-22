sudo apt update
sudo apt install swig -y
curl -LsSf https://astral.sh/uv/install.sh | sh
uv run python train.py --env_name mujoco/walker2d/expert-v0  --ms offline --lr_decay