import os
import gymnasium as gym
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from jamb_env import JambEnv

# Fix: PyTorch's Categorical distribution validates that probs sum to exactly 1.0,
# but floating-point rounding after action masking can violate this by tiny amounts.
torch.distributions.Distribution.set_default_validate_args(False)


def mask_fn(env: gym.Env):
    return env.get_wrapper_attr("action_masks")()

def linear_schedule(initial_value, final_value=2.5e-5):
    """Linear decay from initial_value to final_value."""
    def func(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return func

def train():
    log_dir = "./logs/MaskablePPO_v6/"
    models_dir = "./models/jamb_ppo_v6/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    def make_masked_env():
        env = JambEnv()
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env

    # ── Hardware: 5800X3D (8c/16t), 32GB RAM, RTX 3090 ──
    n_envs = 64
    env = SubprocVecEnv([make_masked_env for _ in range(n_envs)])
    
    # Check for existing checkpoint to resume from
    checkpoint = None
    if os.path.exists(models_dir):
        zips = [f for f in os.listdir(models_dir) if f.endswith(".zip") and "_steps" in f]
        if zips:
            zips.sort(key=lambda x: int(x.split("_steps")[0].rsplit("_", 1)[-1]))
            checkpoint = os.path.join(models_dir, zips[-1])

    if checkpoint:
        print(f"  ⏩ RESUMING from: {os.path.basename(checkpoint)}")
        model = MaskablePPO.load(checkpoint, env=env, device="cuda",
                                  tensorboard_log=log_dir)
        # Restore schedule params that aren't saved in checkpoint
        model.learning_rate = linear_schedule(2.5e-4, 2.5e-5)
        model.ent_coef = 0.01
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            device="cuda",
            learning_rate=linear_schedule(2.5e-4, 2.5e-5),
            n_steps=1024,           # 1024 × 64 = 65,536 steps/rollout (~436 games)
            batch_size=4096,        # 16 mini-batches per epoch
            n_epochs=15,            # 240 gradient updates per rollout (more GPU time)
            gamma=0.999,            # Terminal-only reward
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,          # Moderate exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])
            )
        )
    
    # Save every ~500K env steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),
        save_path=models_dir,
        name_prefix='jamb_ppo_v6'
    )
    
    print("=" * 60)
    print("Jamb v6 Training — Per-Cell Obs + No Keep-All-6")
    print("=" * 60)
    print(f"  Envs:         {n_envs}")
    print(f"  Rollout:      {n_envs * 1024:,} steps ({n_envs * 1024 // 150}+ games)")
    print(f"  Batch size:   4096")
    print(f"  n_epochs:     15")
    print(f"  Network:      pi=[512,512,256]  vf=[512,512,256]")
    print(f"  LR:           2.5e-4 → 2.5e-5 (linear decay)")
    print(f"  Gamma:        0.999")
    print(f"  Entropy:      0.01")
    print(f"  Actions:      {model.action_space}")
    print(f"  Obs:          {model.observation_space}")
    print("=" * 60)
    
    total_timesteps = 100_000_000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback,
                reset_num_timesteps=False)
    
    model.save(os.path.join(models_dir, "jamb_ppo_v6_final"))
    print("Training finished.")

if __name__ == "__main__":
    train()
