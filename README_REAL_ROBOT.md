# Running MolmoAct2 on YAM Arms (Async Inference)

A minimal guide to run the **MolmoAct2** policy on **bimanual YAM arms** using
LeRobot's async inference.

It uses two machines:

- **Server** — a GPU box that runs the policy (`policy_server`) and returns action chunks.
- **Client** — the robot machine that talks to the YAM arms + RealSense cameras
  (`robot_client`), streams observations to the server, and executes the actions.

```
 Client (robot machine, py3.10)                 Server (GPU machine, py3.12)
 ┌───────────────────────────┐                  ┌───────────────────────────┐
 │ YAM follower arms (CAN)    │                  │                           │
 │ RealSense cameras          │ ── observations ─▶  MolmoAct2 policy (CUDA)  │
 │ robot_client               │ ◀── action chunks ─                         │
 └───────────────────────────┘      gRPC          └───────────────────────────┘
```

> Repo: [github.com/SuveenE/lerobot](https://github.com/SuveenE/lerobot)
>
> Branches: client = [`bimanual-yam-arms-support`](https://github.com/SuveenE/lerobot/tree/bimanual-yam-arms-support) (Python **3.10**),
> server = [`molmoact2-yam-async-inference-server`](https://github.com/SuveenE/lerobot/tree/molmoact2-yam-async-inference-server) (Python **3.12**, already updated to use `uv`).

---

## 0. Clone the repo (both machines)

```bash
git clone https://github.com/SuveenE/lerobot.git
cd lerobot
```

Both machines use `uv` — install it first if needed
([uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 1. Server (GPU machine)

Branch `molmoact2-yam-async-inference-server` ships a `uv.lock`, so install is one command.
Requires a CUDA GPU (cu128 wheels, NVIDIA driver ≥ 570.86).

```bash
git checkout molmoact2-yam-async-inference-server && git pull

# Sync the locked env with the async + molmoact2 extras
uv sync --python 3.12 --locked --extra async --extra molmoact2

# Start the policy server (leave it running)
uv run python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8000 \
  --fps=30
```

Note the server machine's IP address — the client connects to `<SERVER_IP>:8000`.

---

## 2. Client (robot machine, Python 3.10)

The client only needs the YAM hardware + camera deps — it never loads the policy,
so the `molmoact2` extra is **not** required here.

### Install (one-time)

You can use `uv` on the client too. This branch has no `uv.lock`, so just create a
3.10 virtualenv with `uv` and install in editable mode.

```bash
git checkout bimanual-yam-arms-support && git pull
git submodule update --init --recursive   # pulls the i2rt submodule

# Create + activate a Python 3.10 env with uv
uv venv --python 3.10
source .venv/bin/activate

# i2rt drives the YAM arms (local submodule), then LeRobot with yam + async + realsense
uv pip install -e ./i2rt
uv pip install -e ".[yam,async,intelrealsense]"
```

<details>
<summary>Prefer pip/conda instead of uv?</summary>

```bash
conda create -y -n lerobot-client python=3.10 && conda activate lerobot-client
pip install -e ./i2rt
pip install -e ".[yam,async,intelrealsense]"
```
</details>

### Step 2a — Start the YAM follower servers (terminal 1)

First reset the CAN interfaces (brings every `can*` interface down then up at
bitrate 1000000) — do this whenever the arms were just powered on or the bus is stuck:

```bash
bash i2rt/scripts/reset_all_can.sh
```

For eval we only need the follower arms (no leader/teaching handles):

```bash
python -m lerobot.scripts.setup_bi_yam_servers --eval
```

This launches the right (port `1234`) and left (port `1235`) follower arm servers.
Keep it running.

### Step 2b — Run the robot client (terminal 2)

Point `--server_address` at the GPU server's IP and port. Replace the RealSense
`serial_number_or_name` values with yours (`lerobot-find-cameras`).

```bash
python -m lerobot.async_inference.robot_client \
  --server_address https://untaken-eskimo-penholder.ngrok-free.dev/act \
  --robot.type bi_yam_follower \
  --robot.left_arm_port 1235 \
  --robot.right_arm_port 1234 \
  --robot.cameras '{
right: {"type": "intelrealsense", "serial_number_or_name": "260322275072", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260322271881", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262522074294", "width": 640, "height": 360, "fps": 30}
}' \
  --task "Put everything into the box." \
  --policy_type molmoact2 \
  --pretrained_name_or_path="" \
  --policy_device cuda \
  --actions_per_chunk 15 \
  --chunk_size_threshold 0.0 \
  --aggregate_fn_name weighted_average \
  --debug_visualize_queue_size True \
  --policy_config_overrides '[
    "--checkpoint_path=allenai/MolmoAct2-BimanualYAM",
    "--norm_tag=yam_dual_molmoact2",
    "--inference_action_mode=continuous",
    "--normalize_gripper=false"
  ]'
```

### Find Serial Numbers for RealSense Cameras
```bash
rs-enumerate-devices | grep -i "serial"
```

---

## Key flags

| Flag | Meaning |
| --- | --- |
| `--server_address` | GPU server `IP:port` the client connects to. |
| `--pretrained_name_or_path=""` | Empty → **HF-original mode**: the MolmoAct2 weights on the Hub are in plain `transformers` format (no LeRobot `config.json`), so the server builds the policy config from scratch entirely out of `--policy_config_overrides`. Set this to a LeRobot-saved checkpoint instead to load its `config.json` directly. |
| `--policy_config_overrides` → `--checkpoint_path` | Where MolmoAct2 weights are loaded from (HF repo id). |

## Notes

- Client and server are **separate environments / machines**: client = Python 3.10,
  server = Python 3.12.
- The `molmoact2` extra lives only on the server branch; the client never instantiates the policy.
- **Data recording is skipped** in this guide. To record eval rollouts, add the
  `--dataset.*` flags — see the example at the top of
  `src/lerobot/async_inference/robot_client.py`

---

## 3. Data Collection (Teleoperation)

Unlike inference, this runs entirely on the client (Python 3.10 env from Step
2) — no GPU server needed. The leader arms (YAM teaching handles) drive the
followers locally while `lerobot-record` saves the episodes.

### Step 3a — Start CAN + all 4 YAM servers (terminal 1)

```bash
bash i2rt/scripts/reset_all_can.sh

# no --eval this time — also starts the leader (teaching handle) servers
python -m lerobot.scripts.setup_bi_yam_servers
```

This launches follower servers on `1234`/`1235` (as before) plus leader
servers on `5001` (right) and `5002` (left).

### Step 3b — Record with `lerobot-record` (terminal 2)

```bash
lerobot-record \
  --robot.type=bi_yam_follower \
  --robot.left_arm_port 1235 \
  --robot.right_arm_port 1234 \
  --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260522274159", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260422272749", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262422072660", "width": 640, "height": 360, "fps": 30}
}' \
  --teleop.type=bi_yam_leader \
  --teleop.left_arm_port 5002 \
  --teleop.right_arm_port 5001 \
  --dataset.repo_id=local/penv2 \
  --dataset.root=/mnt/ssd/Evan/datasets/penv2 \
  --dataset.num_episodes=60 \
  --dataset.episode_time_s=210 \
  --dataset.reset_time_s=300 \
  --dataset.video_encoding_batch_size=60 \
  --dataset.single_task="Grasp the marker and cap then close." \
  --dataset.push_to_hub=false \
  --display_data=false  \
  --robot.record_eef_pose=true
```

`--dataset.root` + `--dataset.push_to_hub=false` keeps everything local — no
Hugging Face account/token needed; `repo_id` is just a local dataset name in
that case, not a real HF namespace.

`--display_data=false` disables the rerun visualization window — recording
still runs fully over CLI, just without the live camera/state GUI, which
removes that overhead.

The gripper is driven by the teaching handle's encoder knob (0 = closed,
1 = open). Add `--robot.record_torques=true` to also save per-arm motor
torques as `observation.{left,right}_torques`.

This produces a LeRobot **v3.0** dataset — the format MolmoAct2's fine-tuning
pipeline expects, so no conversion step is needed. To use it for fine-tuning,
add its `repo_id` to `YAM_BIMANUAL_MOLMOACT2` in the molmoact2 repo's
`experiments/launch_scripts/data_constants.py` and compute/register norm
stats under the `yam_dual_molmoact2` tag.

---

## 4. Fine-tuned MolmoAct2 checkpoint over HTTP (`hqfang/yam_fold_70`)

LeRobot-format MolmoAct2 checkpoints from `hq-fang/lerobot` do not load on the
`molmoact2-yam-async-inference-server` branch (their `config.json` has newer fields), and
that fork's own async server does not list `molmoact2`. So they are served with a small
HTTP server instead, and driven by `scripts/eval_yam_http_policy.py --action_space=joint`
(14-D absolute joint state/actions, no IK). Server and client can share one machine.

### Step 4a — Server env (one-time, Python 3.12, uses ~15 GB VRAM)

```bash
mkdir -p ../molmoact2_server && cd ../molmoact2_server
git clone --branch main https://github.com/hq-fang/lerobot.git && cd lerobot
git checkout 5f8be1af5380d89ba4f645bc69a9c10c632a8025   # commit pinned by the checkpoint
uv sync --python 3.12 --locked --extra molmoact2
uv pip install fastapi uvicorn json-numpy
.venv/bin/hf download hqfang/yam_fold_70 --local-dir ../yam_fold_70

# Base model: config/tokenizer only. The loader wants the 22 GB base weights just to
# overwrite them with the fine-tuned ones, so build them from the checkpoint instead.
.venv/bin/hf download allenai/MolmoAct2-BimanualYAM --revision 8dcbed66f2380e4393189c303ea72488eb9e63c2 \
  --exclude "*.safetensors" --local-dir ../base_lite
.venv/bin/python ../../lerobot_YAM_REAL_ROBOT/scripts/make_molmoact2_base_lite.py ..
```

### Step 4b — Start the policy server (terminal 1)

```bash
cd ../molmoact2_server/lerobot
.venv/bin/python ../../lerobot_YAM_REAL_ROBOT/scripts/host_molmoact2_policy_server.py \
  --checkpoint ../yam_fold_70 --base-model ../base_lite --port 8000
```

### Step 4c — Follower arms (terminal 2) and eval client (terminal 3)

```bash
bash i2rt/scripts/reset_all_can.sh
python -m lerobot.scripts.setup_bi_yam_servers --eval
```

```bash
python scripts/eval_yam_http_policy.py \
  --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260522274159", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260422272749", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262422072660", "width": 640, "height": 360, "fps": 30}
}' \
  --server_url=http://127.0.0.1:8000/act \
  --action_space=joint \
  --task="Pick up the shirt, fold it neatly, and place it on the pile." \
  --actions_per_chunk=15
```

## 5. Evaluate the local BiCAT box checkpoint

This checkpoint is a Praxis `ki_btt_molmo` policy from
`~/Evan/bimanual_tokenizer`, not a LeRobot MolmoAct2 checkpoint. Its
`policy_io.json` declares three RGB cameras (top 640x360, left/right
640x480), a 14-value joint state, and 14-value absolute joint actions.
The BiCAT checkpoint predicts 32 actions and its Praxis queue keeps the
first 16 (`n_action_steps=16`). The bridge returns those 16 actions per
request. LeRobot's `--actions_per_chunk` controls how many of the returned
actions it executes before requesting another prediction: use `16` to match
the training execution horizon, or a smaller value to replan sooner. The
LeRobot client applies joint-space motion limits. Run the bridge in an environment with the
`bimanual_tokenizer` dependencies; the LeRobot `.venv` does not contain Praxis.

First check the checkpoint, BTT tokenizer, and Molmo2-ER processor files without loading
weights or touching the robot:

```bash
cd ~/Evan/lerobot
python scripts/host_bicat_policy_server.py \
  --checkpoint /mnt/ssd/Evan/models/Test/BiCAT/4kstep_box_only/pretrained_model \
  --vlm /mnt/ssd/Evan/models/Test/Molmo2-ER \
  --tokenizer /mnt/ssd/Evan/models/Test/BiCAT_tokenizer/BTT/step50000/pretrained_model \
  --check
```

The BTT tokenizer was downloaded from `Evanxw0715/BiCAT-Test-BOX-ONLY`.
The BiCAT checkpoint contains all Molmo2-ER VLM tensors. The bridge builds
the VLM architecture from the downloaded config/processor files and loads
BiCAT's own weights; no separate Molmo2-ER base weights are required.
Pass these local directories explicitly because the checkpoint's training
paths point to `/scratch/xuanweiliu/`.
The training repo's dependencies were installed with
`cd ~/Evan/bimanual_tokenizer && uv sync --python 3.10 --locked --no-dev`.
Its HTTP bridge also needs `uv pip install json-numpy`.

On this workstation's RTX 5090, the locked PyTorch 2.7.1 CUDA 12.6 wheel
cannot run GPU kernels. After `uv sync`, install the matching CUDA 12.8
build in the `bimanual_tokenizer` environment:

```bash
cd ~/Evan/bimanual_tokenizer
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128'
```

Do not run `uv sync` again before evaluation unless you repeat this CUDA
12.8 step; sync restores the locked CUDA 12.6 packages.

Before using the robot, verify the full checkpoint loads (still no arm
connection):

```bash
cd ~/Evan/bimanual_tokenizer
PYTHONPATH="$PWD/src" .venv/bin/python ~/Evan/lerobot/scripts/host_bicat_policy_server.py \
  --checkpoint /mnt/ssd/Evan/models/Test/BiCAT/4kstep_box_only/pretrained_model \
  --vlm /mnt/ssd/Evan/models/Test/Molmo2-ER \
  --tokenizer /mnt/ssd/Evan/models/Test/BiCAT_tokenizer/BTT/step50000/pretrained_model \
  --device cuda --load-only
```

This CUDA load-only check passed on this workstation. After starting the
follower arm servers, start the policy server with CUDA in a second terminal:

```bash
cd ~/Evan/bimanual_tokenizer
PYTHONPATH="$PWD/src" .venv/bin/python ~/Evan/lerobot/scripts/host_bicat_policy_server.py \
  --checkpoint /mnt/ssd/Evan/models/Test/BiCAT/4kstep_box_only/pretrained_model \
  --vlm /mnt/ssd/Evan/models/Test/Molmo2-ER \
  --tokenizer /mnt/ssd/Evan/models/Test/BiCAT_tokenizer/BTT/step50000/pretrained_model
```

Verify `http://127.0.0.1:8000/act` reports `state_dim=14`,
`action_dim=14`, and camera keys `top,left,right`. With the follower
servers running, evaluate from `~/Evan/lerobot` in a third terminal:

test
```bash
.venv/bin/python scripts/eval_yam_http_policy.py \
  --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --robot.left_arm_max_relative_target=0.05 \
  --robot.right_arm_max_relative_target=0.05 \
  --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260522274159", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260422272749", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262422072660", "width": 640, "height": 360, "fps": 30}
}' \
  --server_url=http://127.0.0.1:8000/act \
  --action_space=joint --reset_server_each_episode=true \
  --task="Put all blocks into the box." \
  --actions_per_chunk=4 --duration_s=10 --max_episodes=1 \
  --max_joint_step_rad=0.03 --max_gripper_step=0.05
```
main
```bash
.venv/bin/python scripts/eval_yam_http_policy.py \
  --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --robot.left_arm_max_relative_target=0.05 \
  --robot.right_arm_max_relative_target=0.05 \
  --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260522274159", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260422272749", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262422072660", "width": 640, "height": 360, "fps": 30}
}' \
  --server_url=http://127.0.0.1:8000/act \
  --action_space=joint --reset_server_each_episode=true \
  --task="Put all the metal parts in the box and lift the box.." \
  --actions_per_chunk=16 --duration_s=120 --max_episodes=1 \
  --max_joint_step_rad=0.03 --max_gripper_step=0.05
```


```bash
.venv/bin/python scripts/eval_yam_http_policy.py \
  --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --robot.left_arm_max_relative_target=0.05 \
  --robot.right_arm_max_relative_target=0.05 \
  --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260522274159", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260422272749", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262422072660", "width": 640, "height": 360, "fps": 30}
}' \
  --server_url=http://127.0.0.1:8000/act \
  --action_space=joint --reset_server_each_episode=true \
  --task="Use the appropriate arm to place the bowl on the plate." \
  --actions_per_chunk=16 --duration_s=400 --max_episodes=5
```

cd ~/Evan/lerobot
.venv/bin/python scripts/eval_yam_http_policy.py \
  --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --robot.left_arm_max_relative_target=0.05 \
  --robot.right_arm_max_relative_target=0.05 \
  --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260522274159", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260422272749", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262422072660", "width": 640, "height": 360, "fps": 30}
}' \
  --server_url=http://127.0.0.1:8000/act \
  --expected_policy_name=bar_btt_molmo \
  --expected_execution_horizon=32 \
  --action_space=joint --reset_server_each_episode=true \
  --chunk_gate_enabled=false \
  --task="Put all bottles in dustbin.." \
  --actions_per_chunk=1 --duration_s=400 --max_episodes=5 \
  --max_joint_step_rad=0.03 --max_gripper_step=0.05




The server accepts only the checkpoint's camera sizes and finite 14-value
states. A synthetic GPU request returned a finite `(16, 14)` action chunk.
The client checks every returned action chunk before commanding a joint.
For the first live run, keep the robot's workspace clear and use one short
episode. Launching the client only connects it; at `[eval] Session ready`,
type `b` then Enter in that terminal to home both arms and start the policy.
Pressing `f` ends the episode and homes
them again. Use the robot's physical stop if motion is unsafe.
After this short trial, set `--actions_per_chunk=16` to match the checkpoint's
16 executed steps and increase `--duration_s` for the full task.
Homing now checks live joint feedback after the four-second interpolation and
continues with small, bounded commands for up to six more seconds if needed.
It prints either `both arms reached home` or `WARNING: home incomplete` with
per-joint errors. An episode will not start if the initial home check fails.
If home is incomplete after `f`, the eval process stays connected: check the
arms, then type `r` and Enter to retry the reset, or `z` and Enter to finish.
When `--max_episodes=1`, a successful home finishes the session automatically.

## 6. Evaluate the autoregressive BiCAT checkpoint

The checkpoint at `/mnt/ssd/Evan/models/Test/BiCAT/pretrained_model` is a
`bar_btt_molmo` policy. Use its matching BTT checkpoint at
`/mnt/ssd/Evan/models/Test/BiCAT_tokenizer/BTT`. This is a different tokenizer
from `BTT/step50000/pretrained_model`. The policy predicts 32 absolute 14-value
joint actions (six joints and a gripper for each arm). Praxis queues the first
16. As in RoboTwin simulation, the client gets a fresh observation before
requesting each single queued action. After action 16, Praxis predicts again.
There is no explicit settling wait between chunks. Use `--action_space=joint`,
`--actions_per_chunk=1`, and `--chunk_gate_enabled=false`; no end-effector IK
is involved. The old chunk gate can hold small first actions at home when
the server returns only one action. The client now bypasses that gate for
single-action responses even if the flag is omitted.

First start the follower servers in terminal 1:

```bash
cd ~/Evan/lerobot
bash i2rt/scripts/reset_all_can.sh
.venv/bin/python -m lerobot.scripts.setup_bi_yam_servers --eval
```

In terminal 2, stop any old policy server on port 8000, then start the AR
checkpoint with the simulation checkout's source and the Praxis model environment.
The `.venv-robotwin` environment lacks the model server's `transformers` and
`json_numpy` dependencies:

```bash
cd /mnt/ssd/Evan/bimanual_tokenizer
PYTHONPATH="$PWD/src" ~/Evan/bimanual_tokenizer/.venv/bin/python ~/Evan/lerobot/scripts/host_bicat_policy_server.py \
  --checkpoint /mnt/ssd/Evan/models/Test/BiCAT/pretrained_model \
  --vlm /mnt/ssd/Evan/models/Test/Molmo2-ER \
  --tokenizer /mnt/ssd/Evan/models/Test/BiCAT_tokenizer/BTT
```

Wait for `Using policy implementation: praxis.policies.bar.variants.BTTMolmoPolicy`
and `bar_btt_molmo policy ready at http://127.0.0.1:8000/act`. In terminal 3:

```bash
cd ~/Evan/lerobot
.venv/bin/python scripts/eval_yam_http_policy.py \
  --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --robot.left_arm_max_relative_target=0.05 \
  --robot.right_arm_max_relative_target=0.05 \
  --robot.cameras='{
right: {"type": "intelrealsense", "serial_number_or_name": "260522274159", "width": 640, "height": 480, "fps": 30},
left: {"type": "intelrealsense", "serial_number_or_name": "260422272749", "width": 640, "height": 480, "fps": 30},
top: {"type": "intelrealsense", "serial_number_or_name": "262422072660", "width": 640, "height": 360, "fps": 30}
}' \
  --server_url=http://127.0.0.1:8000/act \
  --expected_policy_name=bar_btt_molmo \
  --expected_execution_horizon=16 \
  --action_space=joint --reset_server_each_episode=true \
  --chunk_gate_enabled=false \
  --task="Put all the metal parts in the box and lift the box." \
  --actions_per_chunk=1 --duration_s=120 --max_episodes=1 \
  --max_joint_step_rad=0.03 --max_gripper_step=0.05
```

At `Session ready`, type `b` and Enter to home the arms and begin. Type `s`
or `f` and Enter to end and label the episode. The client checks that the
server identifies itself as `bar_btt_molmo` before moving the arms. Change
the task text to match the scene you are evaluating.

To evaluate the downloaded 10000-step AR checkpoint with **predict 32 / execute
32**, restart terminal 2 with
`--checkpoint /mnt/ssd/Evan/models/Test/BiCAT/step_10000/pretrained_model`
and `--execute-steps=32`. Check `curl http://127.0.0.1:8000/act` reports
`"prediction_horizon":32`, `"execution_horizon":32`, and `"chunk_size":1`.
In terminal 3 use `--expected_execution_horizon=32` and keep
`--actions_per_chunk=1`: the server still returns one queued action per fresh
observation. To restore **predict 32 / execute 16**, restart terminal 2 with
`--execute-steps=16` (or omit that flag) and use
`--expected_execution_horizon=16` in terminal 3. Changing only
`--actions_per_chunk` on the client does not change the Praxis queue length.

# SANITY REPLAY DATASET:
```bash
lerobot-replay-bi-yam --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --dataset.repo_id=local/sanity --dataset.root=./datasets/sanity --dataset.episode=0 \
  --method=joint

lerobot-replay-bi-yam --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --dataset.repo_id=local/sanity --dataset.root=./datasets/sanity --dataset.episode=0 \
  --method=eef_absolute

lerobot-replay-bi-yam --robot.left_arm_port=1235 --robot.right_arm_port=1234 \
  --dataset.repo_id=local/sanity --dataset.root=./datasets/sanity --dataset.episode=0 \
  --method=eef_delta
```

# SANITY CHECK CAMERA POSE:
```bash
python sanity_check_cameras.py
```
