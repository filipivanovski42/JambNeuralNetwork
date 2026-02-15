"""
Jamb v5 Parallel Training — 3 independent PPO configs on shared hardware.

Hardware: Ryzen 5800X3D (8c/16t), 32GB RAM, RTX 3090 (24GB VRAM)
  - 12 env subprocesses total (4 per config, leaves 4 threads for OS)
  - ~3GB VRAM total, ~15GB RAM total
"""
import os
import sys
import multiprocessing as mp
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from jamb_env import JambEnv


def mask_fn(env: gym.Env):
    return env.get_wrapper_attr("action_masks")()

def linear_schedule(initial_value, final_value):
    def func(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return func

def make_masked_env():
    env = JambEnv()
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)
    return env


CONFIGS = {
    "a": {
        "name": "A (baseline — small network, moderate entropy)",
        "log_dir": "./logs/MaskablePPO_v5a/",
        "models_dir": "./models/jamb_ppo_v5a/",
        "prefix": "jamb_ppo_v5a",
        "n_envs": 4,
        "lr": (3e-4, 3e-5),
        "ent_coef": 0.01,
        "batch_size": 2048,
        "net_arch": dict(pi=[512, 512, 256], vf=[512, 512, 256]),
    },
    "b": {
        "name": "B (wide network, higher entropy)",
        "log_dir": "./logs/MaskablePPO_v5b/",
        "models_dir": "./models/jamb_ppo_v5b/",
        "prefix": "jamb_ppo_v5b",
        "n_envs": 4,
        "lr": (2.5e-4, 2.5e-5),
        "ent_coef": 0.02,
        "batch_size": 2048,
        "net_arch": dict(pi=[1024, 512], vf=[1024, 512]),
    },
    "c": {
        "name": "C (deep network, low entropy)",
        "log_dir": "./logs/MaskablePPO_v5c/",
        "models_dir": "./models/jamb_ppo_v5c/",
        "prefix": "jamb_ppo_v5c",
        "n_envs": 4,
        "lr": (2e-4, 2e-5),
        "ent_coef": 0.005,
        "batch_size": 1024,
        "net_arch": dict(pi=[512, 512, 512], vf=[512, 512, 512]),
    },
}


def train_config(config_key):
    """Train a single config. Designed to run in a subprocess."""
    cfg = CONFIGS[config_key]
    os.makedirs(cfg["log_dir"], exist_ok=True)
    os.makedirs(cfg["models_dir"], exist_ok=True)

    n_envs = cfg["n_envs"]
    env = SubprocVecEnv([make_masked_env for _ in range(n_envs)])

    lr_init, lr_final = cfg["lr"]

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=cfg["log_dir"],
        device="cuda",
        learning_rate=linear_schedule(lr_init, lr_final),
        n_steps=2048,
        batch_size=cfg["batch_size"],
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=cfg["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=cfg["net_arch"]),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),
        save_path=cfg["models_dir"],
        name_prefix=cfg["prefix"],
    )

    print("=" * 60)
    print(f"Config {cfg['name']}")
    print("=" * 60)
    print(f"  Envs:       {n_envs}")
    print(f"  Rollout:    {n_envs * 2048:,} steps/rollout")
    print(f"  Batch:      {cfg['batch_size']}")
    print(f"  Network:    {cfg['net_arch']}")
    print(f"  LR:         {lr_init} → {lr_final}")
    print(f"  Entropy:    {cfg['ent_coef']}")
    print(f"  Actions:    {model.action_space}")
    print(f"  Obs:        {model.observation_space}")
    print("=" * 60)

    total_timesteps = 50_000_000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)

    final_path = os.path.join(cfg["models_dir"], f"{cfg['prefix']}_final")
    model.save(final_path)
    print(f"Config {config_key} finished. Saved to {final_path}")
    env.close()


def main():
    # Allow running a single config: python train_parallel.py a
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
        for k in keys:
            if k not in CONFIGS:
                print(f"Unknown config '{k}'. Choose from: {list(CONFIGS.keys())}")
                return
        if len(keys) == 1:
            train_config(keys[0])
            return
    else:
        keys = list(CONFIGS.keys())

    # Launch each config in its own process
    processes = []
    for key in keys:
        p = mp.Process(target=train_config, args=(key,), name=f"train_{key}")
        p.start()
        processes.append((key, p))
        print(f"[LAUNCHER] Started config {key} (PID {p.pid})")

    # Wait for all
    for key, p in processes:
        p.join()
        print(f"[LAUNCHER] Config {key} finished (exit code {p.exitcode})")

    print("\n[LAUNCHER] All configs finished!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
