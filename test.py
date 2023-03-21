import envpool
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import numpy as np
import flax 
import jax.numpy as jnp
from typing import Sequence
from huggingface_hub import hf_hub_download
class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x


class Network(nn.Module):
    channelss: Sequence[int] = (16, 32, 32)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channels in self.channelss:
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)

seed = 1
env_id = "UpNDown-v5"
envs = envpool.make(
    "UpNDown-v5",
    env_type="gym",
    num_envs=1,
    episodic_life=True,  # Espeholt et al., 2018, Tab. G.1
    repeat_action_probability=0,  # Hessel et al., 2022 (Muesli) Tab. 10
    noop_max=30,  # Espeholt et al., 2018, Tab. C.1 "Up to 30 no-ops at the beginning of each episode."
    full_action_space=False,  # Espeholt et al., 2018, Appendix G., "Following related work, experts use game-specific action sets."
    max_episode_steps=int(108000 / 4),  # Hessel et al. 2018 (Rainbow DQN), Table 3, Max frames per episode
    reward_clip=True,
    seed=1,
)
envs.num_envs = 1
envs.single_action_space = envs.action_space
envs.single_observation_space = envs.observation_space
repo_url = f"cleanrl/{env_id}-cleanba_ppo_envpool_impala_atari_wrapper-seed1"
model_path = hf_hub_download(
    repo_id=repo_url, filename="cleanba_ppo_envpool_impala_atari_wrapper.cleanrl_model"
)
next_obs = envs.reset()
network = Network()
actor = Actor(action_dim=envs.single_action_space.n)
critic = Critic()
key = jax.random.PRNGKey(seed)
key, network_key, actor_key, critic_key = jax.random.split(key, 4)
network_params = network.init(network_key, np.array([envs.single_observation_space.sample()]))
actor_params = actor.init(actor_key, network.apply(network_params, np.array([envs.single_observation_space.sample()])))
critic_params = critic.init(critic_key, network.apply(network_params, np.array([envs.single_observation_space.sample()])))
# note: critic_params is not used in this script
with open(model_path, "rb") as f:
    (args, (network_params, actor_params, critic_params)) = flax.serialization.from_bytes(
        (None, (network_params, actor_params, critic_params)), f.read()
    )

@jax.jit
def get_action_and_value(
    network_params: flax.core.FrozenDict,
    actor_params: flax.core.FrozenDict,
    next_obs: np.ndarray,
    key: jax.random.PRNGKey,
):
    hidden = network.apply(network_params, next_obs)
    logits = actor.apply(actor_params, hidden)
    # sample action: Gumbel-softmax trick
    # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=logits.shape)
    action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
    return action, key

episodic_returns = []
episodic_return = 0
next_obs = envs.reset()
step = 0
done = False
while not done:
    step += 1
    actions, key = get_action_and_value(network_params, actor_params, next_obs, key)
    next_obs, _, d, infos = envs.step(np.array(actions))
    episodic_return += infos["reward"][0]
    done = sum(infos["terminated"]) + sum(infos["TimeLimit.truncated"]) >= 1
    print(step, infos['TimeLimit.truncated'], infos["terminated"], infos["elapsed_step"], d)

    if step > int(108000 / 4) + 10:
        print("the environment is supposed to be done by now")
        break
print(f"eval_episode={len(episodic_returns)}, episodic_return={episodic_return}")