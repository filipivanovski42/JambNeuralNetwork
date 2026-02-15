"""Quick smoke test for jamb_jax.py"""
import jax
import jax.numpy as jnp

print('Device:', jax.devices())

import jamb_jax as env
print('Actions:', env.TOTAL_ACTIONS)
print('Keep patterns:', env.KEEP_PATTERNS.shape)

# Test reset
key = jax.random.PRNGKey(0)
state, obs = env.reset(key)
print('Obs shape:', obs.shape)
print('Dice:', state.dice_hist)

# Test action mask
mask = env.get_action_mask(state)
print('Valid actions:', jnp.sum(mask))

# Test step
valid_actions = jnp.where(mask, jnp.arange(env.TOTAL_ACTIONS), -1)
action = valid_actions[valid_actions >= 0][0]
key2 = jax.random.PRNGKey(1)
new_state, obs2, reward, done, info = env.step(key2, state, action)
print('After step - reward:', reward, 'done:', done)
print('Obs2 shape:', obs2.shape)

# Test vmap (batch of 16 envs)
keys = jax.random.split(key, 16)
states, obss = jax.vmap(env.reset)(keys)
print('Batch obs shape:', obss.shape)

# Test vmap step
masks = jax.vmap(env.get_action_mask)(states)
print('Batch masks shape:', masks.shape)

# Pick first valid action for each env
def pick_first_valid(mask_row):
    valid = jnp.where(mask_row, jnp.arange(env.TOTAL_ACTIONS), env.TOTAL_ACTIONS)
    return valid.min()

actions = jax.vmap(pick_first_valid)(masks)
step_keys = jax.random.split(jax.random.PRNGKey(2), 16)
new_states, new_obss, rewards, dones, infos = jax.vmap(env.step)(step_keys, states, actions)
print('Batch step done - rewards:', rewards)

print('\n=== ALL SMOKE TESTS PASSED ===')
