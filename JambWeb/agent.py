import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pickle
from flax.training import train_state
import game_logic as env
import sys

class ActorCritic(nn.Module):
    action_dim: int
    actor_layers: list
    critic_layers: list
    activation: str = "relu"

    def setup(self):
        act_fn = getattr(nn, self.activation)
        
        # Actor
        actor_layers = []
        for feat in self.actor_layers:
            actor_layers.append(nn.Dense(feat))
            actor_layers.append(act_fn)
        actor_layers.append(nn.Dense(self.action_dim))
        self.actor_net = nn.Sequential(actor_layers)

        # Critic
        critic_layers = []
        for feat in self.critic_layers:
            critic_layers.append(nn.Dense(feat))
            critic_layers.append(act_fn)
        critic_layers.append(nn.Dense(1))
        self.critic_net = nn.Sequential(critic_layers)

    def __call__(self, x):
        return self.actor_net(x), self.critic_net(x)

class Agent:
    def __init__(self, model_path):
        self.model = ActorCritic(
            action_dim=env.TOTAL_ACTIONS,
            actor_layers=[512, 512, 256],
            critic_layers=[512, 512, 256],
            activation="relu"
        )
        
        # Initialize dummy parameters to load into
        dummy_key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, env.OBS_SIZE))
        self.params = self.model.init(dummy_key, dummy_obs)
        
        # Load weights
        with open(model_path, "rb") as f:
            data = np.load(f, allow_pickle=True)
            # data dict keys: 'params', 'opt_state'
            # We need to reconstruct the params PyTree
            # This is tricky without 'serialization' from flax.
            # Assuming standard save format from train_jax.py which used:
            # flax.serialization.to_bytes(params) ? No, train_jax used:
            # jnp.savez(..., params=params)
            
            # If saved with jnp.savez, likely just arrays. Flax params are nested dicts.
            # Let's hope train_jax used proper serialization or flattened arrays.
            # Checking train_jax.py...
            
            # Code: 
            # ckpt = {'params': runner_state.train_state.params, ...}
            # jnp.savez(path, ckpt)
            
            # JAX/Numpy save/load of Dicts works fine usually.
            loaded_ckpt = dict(data)
            # The 'params' key might be inside a structured array or need reconstruction.
            # Let's assume standard unpickling via allow_pickle=True works for the structured dict.
            # Actually np.load of a dict object stored as 0-d array:
            if 'arr_0' in loaded_ckpt:
                # Compressed dict
                flat_data = loaded_ckpt['arr_0'].item()
                self.params = flat_data['params']
            else:
                # Direct keys if saved with **kwargs?
                # train_jax: jnp.savez(..., **ckpt)
                # So keys should be 'params', 'opt_state', ...
                # But params is a frozen dict. Npz might flatten it.
                # Let's try flexible loading.
                pass
                
        # Re-check train_jax saving method:
        # It saves flat params: param_0, param_1...
        # We need to load them and reconstruct the tree structure.
        
        # 1. Get the trusted structure from dummy initialization
        flat_params, tree_def = jax.tree_util.tree_flatten(self.params)
        
        # 2. Load the flat arrays
        loaded_flat = []
        with np.load(model_path) as data:
            # Check format
            if 'params' in data:
                 # It's the old format or dictionary format
                 self.params = data['params'].item()
                 print("✅ Loaded from 'params' key", file=sys.stderr)
            else:
                 # It's the flat format: param_0, param_1 ...
                 # Ensure we load them in correct order matching tree_flatten
                 # tree_flatten order is deterministic
                 for i in range(len(flat_params)):
                     key = f"param_{i}"
                     if key not in data:
                         raise KeyError(f"Missing parameter {key} in checkpoint")
                     loaded_flat.append(jnp.array(data[key]))
                 
                 # 3. Unflatten
                 self.params = jax.tree_util.tree_unflatten(tree_def, loaded_flat)
                 print(f"✅ Reconstructed PyTree with {len(loaded_flat)} leaves", file=sys.stderr)


        self.apply_jit = jax.jit(self.model.apply)

    def predict(self, state):
        obs = env.get_obs(state)
        mask = env.get_action_mask(state)
        
        # Add batch dim
        obs = jnp.expand_dims(obs, 0)
        
        logits, value = self.apply_jit(self.params, obs)
        logits = logits[0]
        value = value[0, 0]
        
        # Mask
        logits = jnp.where(mask, logits, -1e9)
        probs = jax.nn.softmax(logits)
        
        return probs, value

    def decode_predictions(self, probs, state):
        # Return top 5 valid actions
        mask = env.get_action_mask(state)
        valid_indices = jnp.where(mask)[0]
        
        actions = []
        for idx in valid_indices:
            p = float(probs[idx])
            name = self.get_action_name(int(idx))
            actions.append({'name': name, 'prob': p, 'id': int(idx)})
            
        # Sort
        actions.sort(key=lambda x: x['prob'], reverse=True)
        return actions[:5]

    def get_action_name(self, idx):
        if idx < env.NUM_KEEP_ACTIONS:
            pat = env.KEEP_PATTERNS[idx]
            # pat is histogram [c1, c2, c3, c4, c5, c6]
            kept_dice = []
            for i in range(6):
                count = int(pat[i])
                if count > 0:
                    kept_dice.extend([str(i+1)] * count)
            
            if not kept_dice:
                return "Keep Nothing (Reroll All)"
            return f"Keep {', '.join(kept_dice)}"
        elif idx < env.NUM_KEEP_ACTIONS + env.NUM_SCORE_ACTIONS:
            s_idx = idx - env.NUM_KEEP_ACTIONS
            r, c = s_idx // 4, s_idx % 4
            # We need ROWS/COLS names
            return f"Score {env.ROWS[r]} in {env.COLS[c]}"
        else:
            a_idx = idx - env.NUM_KEEP_ACTIONS - env.NUM_SCORE_ACTIONS
            return f"Announce {env.ROWS[a_idx]}"
