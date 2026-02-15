"""
Jamb RL Training - Pure JAX PPO (GPU-Accelerated)
===================================================
Runs the entire training loop on GPU using JAX.
Can be launched from Windows PowerShell (auto-relays to WSL2)
or directly from WSL2.
"""

import os
import sys
import platform
import subprocess

# ── Auto-relay to WSL2 if running on Windows ─────────────────────────
def _relay_to_wsl():
    """Re-launch this script inside WSL2 for GPU access."""
    # Convert Windows path to WSL path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # C:\Foo\Bar -> /mnt/c/Foo/Bar
    drive = script_dir[0].lower()
    wsl_dir = f'/mnt/{drive}/' + script_dir[3:].replace('\\', '/')
    wsl_script = f'{wsl_dir}/{os.path.basename(__file__)}'

    cmd = [
        'wsl', '-d', 'Ubuntu-22.04', '-u', 'root', '--',
        'bash', '-c',
        f'cd "{wsl_dir}" && python3 "{wsl_script}"'
    ]
    print(f'Relaying to WSL2: python3 {os.path.basename(__file__)}')
    print(f'  WSL path: {wsl_dir}')
    print()
    try:
        proc = subprocess.run(cmd)
        sys.exit(proc.returncode)
    except KeyboardInterrupt:
        sys.exit(0)

if platform.system() == 'Windows':
    _relay_to_wsl()

# ── Imports (only reached inside WSL2) ───────────────────────────────
import time
import json
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import distrax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import NamedTuple
from functools import partial

import jamb_jax as env

# ─── Config ──────────────────────────────────────────────────────────

CONFIG = {
    # Environment
    "NUM_ENVS": 4096,            # Parallel environments on GPU
    "NUM_STEPS": 256,            # Steps per rollout per env

    # Training
    "TOTAL_TIMESTEPS": 4_000_000_000,  # 1B total environment steps (V5)
    "UPDATE_EPOCHS": 4,         # 8 is too few 25 is too many - best result was with 15
    "NUM_MINIBATCHES": 32,       # Minibatches per epoch
    "GAMMA": 1.0,                # No discounting (terminal-reward-only)
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.03,            # 0.03 is too much for this type of game. best result was with 0.02
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,

    # Optimizer
    "LR": 2.5e-4,
    "LR_FINAL": 2.5e-5,
    "ANNEAL_LR": True,

    # Network
    "ACTOR_LAYERS": [512, 512, 256],
    "CRITIC_LAYERS": [512, 512, 256],
    "ACTIVATION": "relu",

    # Logging
    "LOG_DIR": "./logs/jamb_jax_v2_crazyandfast/",
    "MODEL_DIR": "./models/jamb_jax_v2_crazyandfast/",
    "SAVE_INTERVAL": 100,         # Save checkpoint every N updates
    "LOG_INTERVAL": 1,           # Log metrics every N updates
}


# ─── Network ─────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    action_dim: int
    actor_layers: list
    critic_layers: list
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh

        # Actor
        actor = x
        for size in self.actor_layers:
            actor = nn.Dense(size, kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=constant(0.0))(actor)
            actor = act_fn(actor)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                         bias_init=constant(0.0))(actor)

        # Critic
        critic = x
        for size in self.critic_layers:
            critic = nn.Dense(size, kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(critic)
            critic = act_fn(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0),
                        bias_init=constant(0.0))(critic)

        return logits, jnp.squeeze(value, axis=-1)


# ─── Transition Storage ──────────────────────────────────────────────

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mask: jnp.ndarray  # action mask for masked PPO


# ─── Training ────────────────────────────────────────────────────────

def make_train(config):
    """Build the JIT-compiled training function."""

    num_envs = config["NUM_ENVS"]
    num_steps = config["NUM_STEPS"]
    num_updates = config["TOTAL_TIMESTEPS"] // (num_steps * num_envs)
    minibatch_size = (num_envs * num_steps) // config["NUM_MINIBATCHES"]

    print(f"  NUM_UPDATES: {num_updates}")
    print(f"  MINIBATCH_SIZE: {minibatch_size}")
    print(f"  STEPS_PER_UPDATE: {num_steps * num_envs:,}")

    network = ActorCritic(
        action_dim=env.TOTAL_ACTIONS,
        actor_layers=config["ACTOR_LAYERS"],
        critic_layers=config["CRITIC_LAYERS"],
        activation=config["ACTIVATION"],
    )

    def linear_schedule(count):
        # count = number of optimizer steps taken so far
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / num_updates
        return config["LR_FINAL"] + frac * (config["LR"] - config["LR_FINAL"])

    def train(rng):
        # ── Init Network ──
        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros(env.OBS_SIZE)
        params = network.init(init_rng, dummy_obs)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=tx,
        )

        # ── Init Environments ──
        rng, env_rng = jax.random.split(rng)
        env_keys = jax.random.split(env_rng, num_envs)
        env_states, obs = jax.vmap(env.reset)(env_keys)

        # ── Rollout + Update Step ──
        def _update_step(runner_state, update_idx):
            train_state, env_states, obs, rng = runner_state

            # ── Collect Trajectory ──
            def _env_step(carry, _):
                train_state, env_states, obs, rng = carry

                # Get action masks
                masks = jax.vmap(env.get_action_mask)(env_states)

                # Forward pass
                rng, action_rng = jax.random.split(rng)
                logits, values = network.apply(train_state.params, obs)

                # Mask invalid actions (set logits to -1e8)
                masked_logits = jnp.where(masks, logits, -1e8)
                pi = distrax.Categorical(logits=masked_logits)
                actions = pi.sample(seed=action_rng)
                log_probs = pi.log_prob(actions)

                # Step environments
                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, num_envs)
                env_states_new, obs_new, rewards, dones, infos = jax.vmap(env.step)(
                    step_keys, env_states, actions
                )

                # Auto-reset completed envs
                rng, reset_rng = jax.random.split(rng)
                reset_keys = jax.random.split(reset_rng, num_envs)
                reset_states, reset_obs = jax.vmap(env.reset)(reset_keys)

                # Where done, use reset state; else keep stepping
                env_states_final = jax.tree_util.tree_map(
                    lambda r, s: jnp.where(
                        jnp.broadcast_to(dones.reshape(-1, *([1]*(len(s.shape)-1))), s.shape),
                        r, s
                    ),
                    reset_states, env_states_new
                )
                obs_final = jnp.where(dones[:, None], reset_obs, obs_new)

                transition = Transition(
                    done=dones,
                    action=actions,
                    value=values,
                    reward=rewards,
                    log_prob=log_probs,
                    obs=obs,
                    mask=masks,
                )
                carry = (train_state, env_states_final, obs_final, rng)
                return carry, transition

            runner_carry, traj_batch = jax.lax.scan(
                _env_step, (train_state, env_states, obs, rng), None, num_steps
            )
            train_state, env_states, obs, rng = runner_carry

            # ── Calculate GAE ──
            _, last_val = network.apply(train_state.params, obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(carry, transition):
                    gae, next_value = carry
                    delta = (transition.reward
                             + config["GAMMA"] * next_value * (1 - transition.done)
                             - transition.value)
                    gae = (delta
                           + config["GAMMA"] * config["GAE_LAMBDA"]
                           * (1 - transition.done) * gae)
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # ── PPO Update ──
            def _update_epoch(update_state, _):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        logits, values = network.apply(params, traj_batch.obs)
                        # Apply action masking
                        masked_logits = jnp.where(traj_batch.mask, logits, -1e8)
                        pi = distrax.Categorical(logits=masked_logits)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + jnp.clip(
                            values - traj_batch.value,
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(values - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Actor loss (clipped)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae_norm
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"],
                                              1.0 + config["CLIP_EPS"]) * gae_norm
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        # Entropy (only over valid actions)
                        entropy = pi.entropy().mean()

                        total_loss = (loss_actor
                                     + config["VF_COEF"] * value_loss
                                     - config["ENT_COEF"] * entropy)
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)

                batch_size = num_steps * num_envs
                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                    shuffled,
                )
                train_state, loss_info = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )
                return (train_state, traj_batch, advantages, targets, rng), loss_info

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # ── Metrics ──
            # Compute mean reward for completed episodes
            episode_returns = traj_batch.reward  # (num_steps, num_envs)
            episode_dones = traj_batch.done
            # Only count rewards at done steps (terminal reward)
            completed_rewards = jnp.where(episode_dones, episode_returns, 0.0)
            num_completed = jnp.sum(episode_dones)
            mean_return = jnp.sum(completed_rewards) / jnp.maximum(num_completed, 1)

            # Loss metrics (take last epoch, mean over minibatches)
            total_loss, (vf_loss, actor_loss, entropy) = loss_info
            metrics = {
                "mean_return": mean_return,
                "num_completed": num_completed,
                "total_loss": jnp.mean(total_loss[-1]),
                "value_loss": jnp.mean(vf_loss[-1]),
                "actor_loss": jnp.mean(actor_loss[-1]),
                "entropy": jnp.mean(entropy[-1]),
                "lr": linear_schedule(train_state.step),
            }

            runner_state = (train_state, env_states, obs, rng)
            return runner_state, metrics

        # ── Outer scan over updates ──
        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_states, obs, train_rng)

        return runner_state, num_updates

    return train, network


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    config = CONFIG
    os.makedirs(config["LOG_DIR"], exist_ok=True)
    os.makedirs(config["MODEL_DIR"], exist_ok=True)

    steps_per_rollout = config['NUM_ENVS'] * config['NUM_STEPS']
    print("=" * 60)
    print("Jamb JAX Training — Pure GPU PPO")
    print("=" * 60)
    print(f"  Device:       {jax.devices()[0]}")
    print(f"  Envs:         {config['NUM_ENVS']}")
    print(f"  Rollout:      {steps_per_rollout:,} steps ({steps_per_rollout // 150}+ games)")
    print(f"  Batch size:   {steps_per_rollout // config['NUM_MINIBATCHES']}")
    print(f"  n_epochs:     {config['UPDATE_EPOCHS']}")
    print(f"  Network:      pi={config['ACTOR_LAYERS']}  vf={config['CRITIC_LAYERS']}")
    print(f"  LR:           {config['LR']} → {config['LR_FINAL']} (linear decay)")
    print(f"  Gamma:        {config['GAMMA']}")
    print(f"  Entropy:      {config['ENT_COEF']}")
    print(f"  Actions:      Discrete({env.TOTAL_ACTIONS})")
    print(f"  Obs:          Box(0, 1, ({env.OBS_SIZE},), float32)")
    print(f"  Total Steps:  {config['TOTAL_TIMESTEPS']:,}")
    print("=" * 60)

    # Build training function
    train_fn, network = make_train(config)

    # Initialize
    rng = jax.random.PRNGKey(42)
    print("\n⏳ Compiling training functions (this may take a few minutes)...")
    t0 = time.time()

    # We can't scan all updates at once (memory), so we do it in a loop
    # with periodic logging and checkpointing
    runner_state, num_updates = jax.jit(train_fn)(rng)
    train_state, env_states, obs, train_rng = runner_state

    # JIT compile the single update step
    _train_fn_inner, _ = make_train(config)

    @jax.jit
    def _single_update(runner_state, update_idx):
        """Run a single PPO update step."""
        # We need to construct the update step inline
        train_state, env_states, obs, rng = runner_state
        num_envs = config["NUM_ENVS"]
        num_steps = config["NUM_STEPS"]

        # Collect trajectory
        def _env_step(carry, _):
            train_state, env_states, obs, rng = carry
            masks = jax.vmap(env.get_action_mask)(env_states)
            rng, action_rng = jax.random.split(rng)
            logits, values = network.apply(train_state.params, obs)
            masked_logits = jnp.where(masks, logits, -1e8)
            pi = distrax.Categorical(logits=masked_logits)
            actions = pi.sample(seed=action_rng)
            log_probs = pi.log_prob(actions)

            rng, step_rng = jax.random.split(rng)
            step_keys = jax.random.split(step_rng, num_envs)
            env_states_new, obs_new, rewards, dones, infos = jax.vmap(env.step)(
                step_keys, env_states, actions
            )

            rng, reset_rng = jax.random.split(rng)
            reset_keys = jax.random.split(reset_rng, num_envs)
            reset_states, reset_obs = jax.vmap(env.reset)(reset_keys)

            env_states_final = jax.tree_util.tree_map(
                lambda r, s: jnp.where(
                    jnp.broadcast_to(dones.reshape(-1, *([1]*(len(s.shape)-1))), s.shape),
                    r, s
                ),
                reset_states, env_states_new
            )
            obs_final = jnp.where(dones[:, None], reset_obs, obs_new)

            transition = Transition(
                done=dones, action=actions, value=values,
                reward=rewards, log_prob=log_probs, obs=obs, mask=masks,
            )
            return (train_state, env_states_final, obs_final, rng), transition

        (train_state, env_states, obs, rng), traj_batch = jax.lax.scan(
            _env_step, (train_state, env_states, obs, rng), None, num_steps
        )

        # GAE
        _, last_val = network.apply(train_state.params, obs)

        def _get_advantages(carry, transition):
            gae, next_value = carry
            delta = (transition.reward
                     + config["GAMMA"] * next_value * (1 - transition.done)
                     - transition.value)
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_val), last_val),
            traj_batch, reverse=True, unroll=16,
        )
        targets = advantages + traj_batch.value

        # PPO epochs
        def _update_epoch(update_state, _):
            def _update_minibatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    logits, values = network.apply(params, traj_batch.obs)
                    masked_logits = jnp.where(traj_batch.mask, logits, -1e8)
                    pi = distrax.Categorical(logits=masked_logits)
                    log_prob = pi.log_prob(traj_batch.action)
                    value_pred_clipped = traj_batch.value + jnp.clip(
                        values - traj_batch.value, -config["CLIP_EPS"], config["CLIP_EPS"])
                    vl = jnp.square(values - targets)
                    vlc = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(vl, vlc).mean()
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                    la1 = ratio * gae_norm
                    la2 = jnp.clip(ratio, 1-config["CLIP_EPS"], 1+config["CLIP_EPS"]) * gae_norm
                    actor_loss = -jnp.minimum(la1, la2).mean()
                    entropy = pi.entropy().mean()
                    total = actor_loss + config["VF_COEF"]*value_loss - config["ENT_COEF"]*entropy
                    return total, (value_loss, actor_loss, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, perm_rng = jax.random.split(rng)
            batch_size = num_steps * num_envs
            perm = jax.random.permutation(perm_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,)+x.shape[2:]), batch)
            shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((config["NUM_MINIBATCHES"],-1)+x.shape[1:]), shuffled)
            train_state, loss_info = jax.lax.scan(_update_minibatch, train_state, minibatches)
            return (train_state, traj_batch, advantages, targets, rng), loss_info

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
        train_state = update_state[0]
        rng = update_state[-1]

        # Metrics
        completed_rewards = jnp.where(traj_batch.done, traj_batch.reward, 0.0)
        num_completed = jnp.sum(traj_batch.done)
        mean_return = jnp.sum(completed_rewards) / jnp.maximum(num_completed, 1)

        total_loss, (vf_loss, actor_loss, entropy) = loss_info
        metrics = {
            "mean_return": mean_return,
            "num_completed": num_completed,
            "total_loss": jnp.mean(total_loss[-1]),
            "value_loss": jnp.mean(vf_loss[-1]),
            "actor_loss": jnp.mean(actor_loss[-1]),
            "entropy": jnp.mean(entropy[-1]),
        }
        return (train_state, env_states, obs, rng), metrics

    compile_time = time.time() - t0
    print(f"✅ Compilation done in {compile_time:.1f}s")

    # ── TensorBoard Setup (tensorboardX — no PyTorch needed) ──
    # Create unique run directory (PPO_1, PPO_2, ...) like SB3
    run_idx = 1
    while os.path.exists(os.path.join(config["LOG_DIR"], f"PPO_{run_idx}")):
        run_idx += 1
    run_log_dir = os.path.join(config["LOG_DIR"], f"PPO_{run_idx}")
    os.makedirs(run_log_dir, exist_ok=True)

    has_tb = False
    writer = None
    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(run_log_dir)
        has_tb = True
    except ImportError:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(run_log_dir)
            has_tb = True
        except ImportError:
            pass

    if has_tb:
        print(f"📊 TensorBoard: {run_log_dir}")
    else:
        print("⚠️  TensorBoard not available. Install: pip3 install tensorboardX")

    # Always log to CSV as backup
    csv_path = os.path.join(run_log_dir, "training_log.csv")
    csv_file = open(csv_path, "w")
    csv_file.write("update,steps,sps,mean_return,num_completed,total_loss,value_loss,actor_loss,entropy,elapsed\n")

    # ── Training Loop ──
    print(f"\n🚀 Starting training loop ({num_updates} updates)...\n")
    steps_per_update = config["NUM_ENVS"] * config["NUM_STEPS"]
    total_steps = 0
    t_start = time.time()

    for update_idx in range(num_updates):
        t_update = time.time()
        runner_state, metrics = _single_update(runner_state, update_idx)
        # Block until computation is done (for timing)
        jax.block_until_ready(metrics)
        dt = time.time() - t_update

        total_steps += steps_per_update
        sps = steps_per_update / dt

        # Extract metrics (move from device)
        m = {k: float(v) for k, v in metrics.items()}

        # Log
        if update_idx % config["LOG_INTERVAL"] == 0:
            elapsed = time.time() - t_start
            current_lr = config["LR_FINAL"] + (1.0 - update_idx / num_updates) * (config["LR"] - config["LR_FINAL"])
            n_updates_done = (update_idx + 1) * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]

            # SB3-style table output
            rows = [
                ("", "rollout/", ""),
                ("", "   ep_rew_mean", f"{m['mean_return']:.3g}"),
                ("", "   games_completed", f"{m['num_completed']:.0f}"),
                ("", "time/", ""),
                ("", "   fps", f"{sps:.0f}"),
                ("", "   iterations", f"{update_idx + 1}"),
                ("", "   time_elapsed", f"{elapsed:.0f}"),
                ("", "   total_timesteps", f"{total_steps}"),
                ("", "train/", ""),
                ("", "   entropy_loss", f"{-m['entropy']:.3g}"),
                ("", "   learning_rate", f"{current_lr:.6g}"),
                ("", "   loss", f"{m['total_loss']:.4g}"),
                ("", "   n_updates", f"{n_updates_done}"),
                ("", "   policy_gradient_loss", f"{m['actor_loss']:.4g}"),
                ("", "   value_loss", f"{m['value_loss']:.4g}"),
            ]
            # Compute column widths
            col1_w = max(len(r[1]) for r in rows) + 2
            col2_w = max(len(r[2]) for r in rows) + 2
            table_w = col1_w + col2_w + 5  # | + space + col1 + space + | + space + col2 + space + |
            print("-" * table_w)
            for _, key, val in rows:
                print(f"| {key:<{col1_w}}| {val:>{col2_w}}|")
            print("-" * table_w)

            if has_tb:
                # Match SB3 metric names
                writer.add_scalar("rollout/ep_rew_mean", m["mean_return"], total_steps)
                writer.add_scalar("rollout/ep_len_mean", m["num_completed"], total_steps)
                writer.add_scalar("train/loss", m["total_loss"], total_steps)
                writer.add_scalar("train/value_loss", m["value_loss"], total_steps)
                writer.add_scalar("train/policy_gradient_loss", m["actor_loss"], total_steps)
                writer.add_scalar("train/entropy_loss", -m["entropy"], total_steps)
                writer.add_scalar("train/learning_rate", current_lr, total_steps)
                writer.add_scalar("time/fps", sps, total_steps)
                writer.add_scalar("time/total_timesteps", total_steps, total_steps)
                writer.flush()

            # CSV logging (always)
            csv_file.write(f"{update_idx},{total_steps},{sps:.0f},{m['mean_return']:.2f},"
                          f"{m['num_completed']:.0f},{m['total_loss']:.6f},"
                          f"{m['value_loss']:.6f},{m['actor_loss']:.6f},"
                          f"{m['entropy']:.4f},{elapsed:.1f}\n")
            csv_file.flush()

        # Save checkpoint
        if update_idx > 0 and update_idx % config["SAVE_INTERVAL"] == 0:
            ckpt_path = os.path.join(config["MODEL_DIR"], f"ckpt_{total_steps}.npz")
            train_state_cpu = runner_state[0]
            params_flat = jax.tree_util.tree_leaves(train_state_cpu.params)
            # Save as numpy arrays
            save_dict = {}
            for i, p in enumerate(params_flat):
                save_dict[f"param_{i}"] = np.array(p)
            np.savez(ckpt_path, **save_dict)
            # Also save tree structure
            tree_struct = jax.tree_util.tree_structure(train_state_cpu.params)
            with open(ckpt_path.replace('.npz', '_tree.json'), 'w') as f:
                json.dump(str(tree_struct), f)
            print(f"  💾 Checkpoint saved: {ckpt_path}")

    # ── Final Save ──
    final_path = os.path.join(config["MODEL_DIR"], "jamb_jax_v1_final.npz")
    train_state_cpu = runner_state[0]
    params_flat = jax.tree_util.tree_leaves(train_state_cpu.params)
    save_dict = {}
    for i, p in enumerate(params_flat):
        save_dict[f"param_{i}"] = np.array(p)
    np.savez(final_path, **save_dict)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Total steps:  {total_steps:,}")
    print(f"  Total time:   {total_time:.1f}s")
    print(f"  Avg SPS:      {total_steps / total_time:,.0f}")
    print(f"  Model saved:  {final_path}")
    print(f"{'='*60}")

    if has_tb:
        writer.close()


if __name__ == "__main__":
    main()
