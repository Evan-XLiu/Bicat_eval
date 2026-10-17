#!/usr/bin/env python3
"""Serve a Praxis BiCAT or OAT checkpoint to eval_yam_http_policy.py over local HTTP.

Run with the Python environment of ~/Evan/bimanual_tokenizer, with that repo's
src directory on PYTHONPATH. Use --check first; it never loads model weights.
"""

import argparse
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

CAMERAS = ("top", "left", "right")
STATE_KEY = "observation.state_joint_angles"
ACTION_KEY = "action_joint_angles"


def local_training_path(value: str) -> Path:
    """Translate the training machine's /scratch/xuanweiliu checkout prefix."""
    prefix = "/scratch/xuanweiliu/"
    if value.startswith(prefix):
        value = "/home/workstation1/Evan/" + value[len(prefix):]
    return Path(value).expanduser().resolve()


def check_external_assets(vlm: Path, tokenizer: Path) -> None:
    for label, path in (("Molmo2-ER metadata", vlm), ("action tokenizer", tokenizer)):
        if not path.is_dir():
            raise ValueError(f"{label} directory does not exist: {path}")
        if not (path / "config.json").is_file():
            raise ValueError(f"{label} is missing config.json: {path}")
    for name in ("tokenizer.json", "processor_config.json", "preprocessor_config.json"):
        if not (vlm / name).is_file():
            raise ValueError(f"Molmo2-ER metadata is missing {name}: {vlm}")
    if not (tokenizer / "model.safetensors").is_file():
        raise ValueError(f"Action tokenizer is missing model weights: {tokenizer}")
    for name in ("norm_stats.pt", "policy_io.json"):
        if not (tokenizer / name).is_file():
            raise ValueError(f"Action tokenizer is missing {name}: {tokenizer}")


def checkpoint_contract(checkpoint: Path) -> dict:
    required = ("config.json", "policy_io.json", "norm_stats.pt", "model.safetensors")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"Checkpoint is missing {missing}: {checkpoint}")
    config = json.loads((checkpoint / "config.json").read_text())
    io = json.loads((checkpoint / "policy_io.json").read_text())
    expected = [STATE_KEY, *(f"observation.images.{key}" for key in CAMERAS)]
    found = [item["key"] for item in io["obs_features"]]
    if config.get("name") not in ("ki_btt_molmo", "ki_oat_molmo", "bar_btt_molmo") or found != expected:
        raise ValueError(f"Unexpected policy or observation layout: {config.get('name')}, {found}")
    if io["obs_features"][0]["shape"] != [14] or io["action_feature"]["key"] != ACTION_KEY:
        raise ValueError("Checkpoint is not a 14-D joint policy")
    if io["action_feature"]["shape"] != [14] or io.get("n_obs_steps") != 1:
        raise ValueError("Unexpected action dimension or observation history")
    if io.get("fps") != 30 or io.get("language_key") != "task":
        raise ValueError("Unexpected policy timing or language key")
    for feature, size in zip(io["obs_features"][1:], ((3, 360, 640), (3, 480, 640), (3, 480, 640))):
        if tuple(feature["shape"]) != size:
            raise ValueError(f"Unexpected camera shape for {feature['key']}: {feature['shape']}")
    return config


def build_app(checkpoint: Path, vlm: Path, tokenizer: Path, device: str, execute_steps: int = 16):
    import json_numpy
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    from unittest.mock import patch
    from praxis.policies.eval_adapter import PolicyEvalAdapter
    from praxis.policies.factory import available_policies, get_policy_class, register_policy

    config = checkpoint_contract(checkpoint)
    if execute_steps not in (16, 32):
        raise ValueError("--execute-steps must be 16 or 32")
    if execute_steps == 32 and config["name"] != "bar_btt_molmo":
        raise ValueError("--execute-steps=32 is supported only for the AR BTT Molmo policy")
    if config.get("action_horizon") != 32 or config.get("n_action_steps") != 16:
        raise ValueError("Expected a checkpoint trained with a 32-step horizon and 16 execution steps")
    # Override only the rollout queue length. Model weights and the 32-step
    # action decoder remain exactly as saved in the checkpoint.
    config["n_action_steps"] = execute_steps
    config["vlm_model_name"] = str(vlm)
    config["action_tokenizer"]["pretrained_path"] = str(tokenizer)
    if config["name"] == "bar_btt_molmo" and "bar_btt_molmo" not in available_policies():
        register_policy("bar_btt_molmo", "bar_btt_molmo_policy:BTTMolmoPolicy")
    policy_cls = get_policy_class(config["name"])
    print(f"Using policy implementation: {policy_cls.__module__}.{policy_cls.__name__}", flush=True)
    # The policy export contains all VLM tensors. Praxis normally loads the
    # original Molmo2-ER weights before overlaying the policy checkpoint. Build
    # its architecture from config instead, then check all non-codec weights.
    def construct_vlm(model_name, **kwargs):
        vlm_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        vlm_config._attn_implementation = "sdpa"
        dtype = kwargs.get("dtype", torch.bfloat16)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            return AutoModelForImageTextToText.from_config(vlm_config, trust_remote_code=True)
        finally:
            torch.set_default_dtype(previous_dtype)

    # Praxis exports the frozen action tokenizer weights separately, so the policy file is
    # intentionally missing action_tokenizer.* tensors. The tokenizer above is
    # loaded from its own checkpoint before the policy weights are applied.
    with patch.object(AutoModelForImageTextToText, "from_pretrained", side_effect=construct_vlm):
        policy = policy_cls.from_pretrained(str(checkpoint), config=config, strict=False)
    from safetensors import safe_open
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as handle:
        saved_keys = set(handle.keys())
    model_keys = set(policy.state_dict())
    missing = model_keys - saved_keys
    extra = saved_keys - model_keys
    if any(not key.startswith("action_tokenizer.") for key in missing) or extra:
        raise RuntimeError(f"Policy weight mismatch: missing={sorted(missing)[:8]}, extra={sorted(extra)[:8]}")
    if policy.io.action_dim != 14 or policy.config.n_action_steps != execute_steps:
        raise ValueError("Loaded policy has an unexpected action contract")
    if policy._action_queue.maxlen != execute_steps:
        raise ValueError("Policy action queue length does not match --execute-steps")
    class Float32PolicyEvalAdapter(PolicyEvalAdapter):
        def _predict_raw_action(self, observations, *, policy_kwargs):
            # Praxis converts this tensor directly to NumPy. The BiCAT
            # checkpoint produces bfloat16 actions under CUDA autocast, which
            # NumPy cannot represent; cast only the output to float32.
            return super()._predict_raw_action(observations, policy_kwargs=policy_kwargs).float()

    adapter = Float32PolicyEvalAdapter(policy, device=device)
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body, status=200):
            data = json_numpy.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/act":
                self.respond({"error": "unknown path"}, 404)
                return
            self.respond({"status": "ok", "policy_name": config["name"],
                          "checkpoint": str(checkpoint), "state_dim": 14,
                          "action_dim": 14, "camera_keys": list(CAMERAS),
                          "chunk_size": 1 if config["name"] == "bar_btt_molmo" else 16,
                          "prediction_horizon": 32 if config["name"] == "bar_btt_molmo" else None,
                          "execution_horizon": execute_steps})

        def do_POST(self):
            if self.path not in ("/act", "/reset"):
                self.respond({"error": "unknown path"}, 404)
                return
            if self.path == "/reset":
                with lock:
                    adapter.reset()
                self.respond({"status": "ok"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 16_000_000:
                    raise ValueError("Invalid request size")
                payload = json_numpy.loads(self.rfile.read(size).decode())
                state = np.asarray(payload["state"], dtype=np.float32)
                if state.shape != (14,) or not np.isfinite(state).all():
                    raise ValueError("Expected finite 14-D joint state")
                observation = {STATE_KEY: state, "task": str(payload["instruction"])}
                for camera, shape in zip(CAMERAS, ((360, 640, 3), (480, 640, 3), (480, 640, 3))):
                    image = np.asarray(payload[camera])
                    if image.shape != shape or image.dtype != np.uint8:
                        raise ValueError(f"{camera}: expected uint8 image with shape {shape}")
                    # The HTTP client sends HWC RGB frames; Praxis PolicyIO
                    # expects CHW arrays for each image feature.
                    observation[f"observation.images.{camera}"] = np.transpose(image, (2, 0, 1))
                started = time.perf_counter()
                # Molmo2's SDPA vision path expects CUDA autocast when the
                # checkpoint uses bfloat16; without it, value is promoted to
                # float32 while query and key remain bfloat16.
                with lock, torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
                ):
                    # RoboTwin requests one action per fresh observation. Praxis
                    # caches the first 16 of 32 predicted rows internally and
                    # generates a new chunk only after that queue is consumed.
                    count = 1 if config["name"] == "bar_btt_molmo" else 16
                    actions = np.stack([adapter.act([observation])[0] for _ in range(count)])
                if actions.shape != (count, 14) or not np.isfinite(actions).all():
                    raise ValueError(f"Policy returned invalid actions: {actions.shape}")
                self.respond({"actions": actions.astype(np.float32),
                              "dt_ms": (time.perf_counter() - started) * 1000})
            except Exception as exc:
                traceback.print_exc()
                self.respond({"error": repr(exc)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, help="Local Molmo2-ER model directory")
    parser.add_argument("--tokenizer", type=Path, help="Local pretrained action tokenizer directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute-steps", type=int, choices=(16, 32), default=16,
                        help="Number of predicted actions to execute before replanning (default: 16)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true", help="Validate paths and contract without loading weights")
    parser.add_argument("--load-only", action="store_true", help="Load policy and exit without serving or moving hardware")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    try:
        config = checkpoint_contract(checkpoint)
        if args.execute_steps == 32 and config["name"] != "bar_btt_molmo":
            raise ValueError("--execute-steps=32 is supported only for the AR BTT Molmo policy")
        vlm = args.vlm.expanduser().resolve() if args.vlm else local_training_path(config["vlm_model_name"])
        tokenizer = (args.tokenizer.expanduser().resolve() if args.tokenizer else
                     local_training_path(config["action_tokenizer"]["pretrained_path"]))
        check_external_assets(vlm, tokenizer)
        tokenizer_name = json.loads((tokenizer / "config.json").read_text()).get("name")
        expected_tokenizer = config["action_tokenizer"]["name"]
        if tokenizer_name != expected_tokenizer:
            raise ValueError(
                f"Checkpoint requires {expected_tokenizer!r} tokenizer, got {tokenizer_name!r}: {tokenizer}"
            )
        if args.check:
            print(f"Checkpoint contract and assets OK: {checkpoint}; execute_steps={args.execute_steps}")
            return
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    handler = build_app(checkpoint, vlm, tokenizer, args.device, args.execute_steps)
    if args.load_only:
        print("Policy loaded successfully", flush=True)
        return
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"{config['name']} policy ready at http://{args.host}:{args.port}/act", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
