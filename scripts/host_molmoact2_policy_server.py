#!/usr/bin/env python
"""HTTP policy server for a LeRobot-format MolmoAct2 bimanual YAM checkpoint
(e.g. `hqfang/yam_fold_70`), speaking the `POST /act` contract of
`scripts/eval_yam_http_policy.py --action_space=joint`.

This does NOT run in this repo's Python 3.10 client env: the checkpoint needs the MolmoAct2
policy code from `hq-fang/lerobot` (pinned commit in the checkpoint's README), so run it
with that checkout's Python 3.12 env. The script itself only imports the checkpoint's own
`inference.py` (its validated loader) plus fastapi/uvicorn/json_numpy.

    GET  /act -> {"state_dim": 14, "action_dim": 14, "camera_keys": ["top", "left", "right"], ...}
    POST /act    json_numpy body {"top", "left", "right": HWC uint8 RGB, "instruction": str,
                                  "state": (14,) float}
              -> {"actions": (30, 14) float32, "dt_ms": float}

State and action rows are ABSOLUTE joint targets: left [6 joints (rad), gripper (0-1)] ++
right [same]. Every request generates a fresh 30-row chunk at 30 Hz; the client decides how
many rows to execute before asking again.

Usage (from the hq-fang/lerobot checkout):

```shell
.venv/bin/python /path/to/scripts/host_molmoact2_policy_server.py \
  --checkpoint ../yam_fold_70 --port 8000
```
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, Response

CAM_KEYS = ["top", "left", "right"]
STATE_DIM = 14
GRIPPER_IDX = [6, 13]


def build_app(checkpoint: Path, device: str, base_model: str | None) -> FastAPI:
    # The checkpoint ships its own loader; use it so loading matches the export's validation.
    sys.path.insert(0, str(checkpoint))
    from inference import load_policy, observation

    policy, pre, post = load_policy(str(checkpoint), device, base_model)
    policy.reset()
    lock = threading.Lock()
    app = FastAPI()

    # NOTE: json_numpy.patch() is deliberately not used (it monkeypatches stdlib `json`
    # process-wide); bodies are encoded/decoded explicitly instead.
    def reply(obj: dict, status: int = 200) -> Response:
        return Response(json_numpy.dumps(obj), status_code=status, media_type="application/json")

    @app.get("/act")
    def health() -> Response:
        return reply(
            {
                "status": "ok",
                "checkpoint": str(checkpoint),
                "state_dim": STATE_DIM,
                "action_dim": STATE_DIM,
                "camera_keys": CAM_KEYS,
                "chunk_size": 30,
            }
        )

    @app.post("/act")
    async def act(request: Request) -> Response:
        try:
            payload = json_numpy.loads(await request.body())
            state = np.asarray(payload["state"], dtype=np.float32).copy()
            # The exported processor rejects grippers outside [0, 1]; the real encoder can
            # overshoot by a hair at either end stop.
            state[GRIPPER_IDX] = np.clip(state[GRIPPER_IDX], 0.0, 1.0)
            raw = observation(*(payload[k] for k in CAM_KEYS), state, payload["instruction"])
            t0 = time.perf_counter()
            with lock, torch.inference_mode():
                normalized = policy.predict_action_chunk(pre(raw), inference_action_mode="continuous")
                actions = post(normalized)[0].detach().float().cpu().numpy()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if actions.shape[1] != STATE_DIM or not np.isfinite(actions).all():
                raise RuntimeError(f"invalid prediction: shape={actions.shape}")
            return reply({"actions": actions.astype(np.float32), "dt_ms": dt_ms})
        except Exception as e:  # noqa: BLE001 -- the client surfaces `error` and aborts the episode
            return reply({"error": repr(e)})

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Local dir of the downloaded checkpoint")
    p.add_argument("--base-model", help="Optional local copy of allenai/MolmoAct2-BimanualYAM")
    p.add_argument("--device", default="cuda")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if not (checkpoint / "inference.py").is_file():
        p.error(f"{checkpoint} is not a downloaded checkpoint dir (no inference.py)")
    uvicorn.run(build_app(checkpoint, args.device, args.base_model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
