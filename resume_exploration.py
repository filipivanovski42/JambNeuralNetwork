import os
import glob
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from jamb_env import JambEnv

def mask_fn(env):
    return env.get_wrapper_attr("action_masks")()

def make_masked_env():
    env = JambEnv()
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)
    return env

def resume_with_exploration():
    # Paths
    models_dir = "./models/jamb_ppo_v3/"
    log_dir = "./logs/MaskablePPO_v3/"
    
    # Find latest checkpoint
    checkpoints = glob.glob(os.path.join(models_dir, "*.zip"))
    if not checkpoints:
        print("No checkpoints found to resume!")
        return
        
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"🚀 Found Latest Checkpoint: {latest_checkpoint}")
    
    # Environment Setup
    n_envs = 32
    print(f"🔧 Creating {n_envs} environments...")
    env = SubprocVecEnv([make_masked_env for _ in range(n_envs)])
    
    # Load Model
    print(f"🧠 Loading model for STABILIZATION (lowering entropy)...")
    model = MaskablePPO.load(latest_checkpoint, env=env, tensorboard_log=log_dir, device="cuda")
    
    # CONVERGENCE PHASE:
    # Lower entropy to let agent exploit its findings.
    model.ent_coef = 0.01
    
    # Callback
    # Save every 1,000,000 steps (31250 * 32 = 1,000,000)
    checkpoint_callback = CheckpointCallback(
        save_freq=31250, 
        save_path=models_dir,
        name_prefix='jamb_ppo_v3'
    )
    
    print("📉 Starting Stabilization Phase. Score should recover and improve.")
    total_timesteps = 100_000_000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, reset_num_timesteps=False)

if __name__ == "__main__":
    resume_with_exploration()
