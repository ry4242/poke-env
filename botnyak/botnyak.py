import json
import os
from typing import Any, Awaitable

import numpy as np
import supersuit as ss
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from poke_env.battle import AbstractBattle
from poke_env.data import GenData
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, Player

BATTLE_FORMAT = "gen9swserandombattle"
N_FEATURES = 12
_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "botnyak_model")
STATS_PATH = os.path.join(_DIR, "botnyak_stats.json")
WINRATES_PATH = os.path.join(_DIR, "winrates.txt")


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[64, 64],
            features_extractor_class=FeaturesExtractor,
        )

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        mask = torch.where(self._mask == 1, 0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits + mask)


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]


class PolicyPlayer(Player):
    policy: ActorCriticPolicy | None

    def __init__(
        self, policy: ActorCriticPolicy | None = None, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def choose_move(
        self, battle: AbstractBattle
    ) -> BattleOrder | Awaitable[BattleOrder]:
        if battle.wait:
            return DefaultBattleOrder()
        obs = self.embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle))
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(
                    obs, device=self.policy.device
                ).unsqueeze(0),
                "action_mask": torch.as_tensor(
                    mask, device=self.policy.device
                ).unsqueeze(0),
            }
            action, _, _ = self.policy.forward(obs_dict)
        action = action.cpu().numpy()[0]
        return SinglesEnv.action_to_order(action, battle)

    @staticmethod
    def embed_battle(battle: AbstractBattle):
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(battle.gen).type_chart,
                )
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )
        our_hp = (
            battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
        )
        opp_hp = (
            battle.opponent_active_pokemon.current_hp_fraction
            if battle.opponent_active_pokemon
            else 0.0
        )
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
                [our_hp, opp_hp],
            ],
            dtype=np.float32,
        )


class SelfPlayEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }
    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=30.0,
        )

    def embed_battle(self, battle: AbstractBattle):
        return PolicyPlayer.embed_battle(battle)


def write_winrates(stats):
    wins = stats["total_wins"]
    battles = stats["total_battles"]
    rate = f"{wins / battles:.1%}" if battles else "N/A"
    lines = [
        "Botnyak Win Rates",
        "=================",
        f"Win Rate:   {rate:>8}",
        f"Wins:       {wins:>8,}",
        f"Battles:    {battles:>8,}",
        f"Sessions:   {stats['sessions']:>8,}",
        f"Steps:      {stats['total_steps']:>8,}",
    ]
    open(WINRATES_PATH, "w").write("\n".join(lines) + "\n")


def train():
    stats = (
        json.loads(open(STATS_PATH).read())
        if os.path.exists(STATS_PATH)
        else {"sessions": 0, "total_steps": 0, "total_wins": 0, "total_battles": 0}
    )

    num_envs = 2
    env = SelfPlayEnv(battle_format=BATTLE_FORMAT, log_level=40, open_timeout=None)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env,
        num_vec_envs=num_envs,
        num_cpus=num_envs,
        base_class="stable_baselines3",
    )

    if os.path.exists(MODEL_PATH + ".zip"):
        ppo = PPO.load(MODEL_PATH, env=vec_env)
        print(f"Resuming from session {stats['sessions']} ({stats['total_steps']:,} steps trained so far)")
    else:
        ppo = PPO(
            MaskedActorCriticPolicy,
            vec_env,
            learning_rate=3e-4,
            n_steps=3072 // (2 * num_envs),
            batch_size=128,
            gamma=0.99,
            ent_coef=0.01,
            device="cpu",
        )

    ppo.learn(98_304)
    vec_env.close()

    ppo.save(MODEL_PATH)
    stats["sessions"] += 1
    stats["total_steps"] += 98_304
    stats["total_wins"] += env.agent1.n_won_battles
    stats["total_battles"] += env.agent1.n_finished_battles
    open(STATS_PATH, "w").write(json.dumps(stats, indent=2))
    write_winrates(stats)
    print(f"Session {stats['sessions']} complete — {stats['total_steps']:,} steps trained total")


if __name__ == "__main__":
    import sys
    if "--reset-wr" in sys.argv:
        stats = (
            json.loads(open(STATS_PATH).read())
            if os.path.exists(STATS_PATH)
            else {"sessions": 0, "total_steps": 0, "total_wins": 0, "total_battles": 0}
        )
        stats["total_wins"] = 0
        stats["total_battles"] = 0
        open(STATS_PATH, "w").write(json.dumps(stats, indent=2))
        write_winrates(stats)
        print("Win rate stats reset.")
    train()
