import json
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
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
from poke_env.concurrency import handle_threaded_coroutines
from poke_env.player import BattleOrder, DefaultBattleOrder, Player, SimpleHeuristicsPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

BATTLE_FORMAT = "gen9swserandombattle"
N_FEATURES = 12
_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.join(_DIR, "stats")
os.makedirs(STATS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(STATS_DIR, "botnyak_model")
STATS_PATH = os.path.join(STATS_DIR, "botnyak_stats.json")
WINRATES_TRAIN_PATH = os.path.join(STATS_DIR, "winrates_train.txt")
WINRATES_SERVE_PATH = os.path.join(STATS_DIR, "winrates_serve.txt")


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


def _get_species_name(mon) -> str:
    species = mon.species
    item = mon.item or ""
    moves = list(mon.moves.keys())

    s = species.lower()

    if s.startswith("pikachu-") or (s.startswith("pikachu") and s != "pikachu"):
        return "pikachu"
    elif s.startswith("unown-") or (s.startswith("unown") and s != "unown"):
        return "unown"
    elif s.startswith("basculin-") or (s.startswith("basculin") and s != "basculin"):
        return "basculin"
    elif s.startswith("sawsbuck-") or (s.startswith("sawsbuck") and s != "sawsbuck"):
        return "sawsbuck"
    elif s.startswith("vivillon-") or (s.startswith("vivillon") and s != "vivillon"):
        return "vivillon"
    elif s.startswith("florges-") or (s.startswith("florges") and s != "florges"):
        return "florges"
    elif s.startswith("furfrou-") or (s.startswith("furfrou") and s != "furfrou"):
        return "furfrou"
    elif s.startswith("minior-") or (s.startswith("minior") and s != "minior"):
        return "minior"
    elif s.startswith("toxtricity-") or (s.startswith("toxtricity") and s != "toxtricity"):
        return "toxtricity"
    elif s.startswith("tatsugiri-") or (s.startswith("tatsugiri") and s != "tatsugiri"):
        return "tatsugiri"
    elif s.startswith("alcremie-") or (s.startswith("alcremie") and s != "alcremie"):
        return "alcremie"
    elif s.startswith("deerling-") or (s.startswith("deerling") and s != "deerling"):
        return "deerling"
    elif s.startswith("flabébé-") or (s.startswith("flabébé") and s != "flabébé"):
        return "flabébé"
    elif s.startswith("botnyak-") or (s.startswith("botnyak") and s != "botnyak"):
        return "botnyak"
    elif s in ("gastrodon-east", "gastrodoneast"):
        return "gastrodon"
    elif s in ("magearna-original", "magearnaoriginal"):
        return "magearna"
    elif s in ("genesect-douse", "genesectdouse"):
        return "genesect"
    elif s in ("dudunsparce-three-segment", "dudunsparse-three-segment", "dudunsparce-threesegment", "dudunsparcethreesegment"):
        return "dudunsparce"
    elif s in ("maushold-four", "mausholdfourfamily", "mausholdfour"):
        return "maushold"
    elif s in ("greninja-bond", "greninjabond"):
        return "greninja"
    elif s in ("keldeo-resolute", "keldeoresolute"):
        return "keldeo"
    elif s in ("zarude-dada", "zarudedada"):
        return "zarude"
    elif s in ("polteageist-antique", "polteageistantique"):
        return "polteageist"
    elif s in ("sinistcha-masterpiece", "sinistchamasterpiece"):
        return "sinistcha"
    elif s in ("squawkabilly-blue", "squawkabillyblue"):
        return "squawkabilly"
    elif s in ("squawkabilly-white", "squawkabillywhite"):
        return "squawkabillyyellow"
    elif s in ("poltchageist-artisan", "poltchageistartisan"):
        return "poltchageist"
    elif s in ("shellos-east", "shelloseast"):
        return "shellos"
    elif s in ("sinistea-antique", "sinisteaantique"):
        return "sinistea"
    elif s == "zacian" and item == "rustedsword":
        return "zaciancrowned"
    elif s == "zamazenta" and item == "rustedshield":
        return "zamazentacrowned"
    elif s == "kyogre" and item == "blueorb":
        return "kyogreprimal"
    elif s == "groudon" and item == "redorb":
        return "groudonprimal"
    elif s == "rayquaza" and "dragonascent" in moves:
        return "rayquazamega"
    elif s == "castform" and item == "whirligig":
        return "castformwhirly"
    elif s in ("snover-lowland", "snowerlowland") and item == "weathervane":
        return "snover"
    elif s == "snover" and item == "weathervane":
        return "snoverlowland"
    elif s in ("abomasnow-lowland", "abomasnowlowland") and item == "weathervane":
        return "abomasnow"
    elif s == "abomasnow" and item == "weathervane":
        return "abomasnowlowland"
    elif s == "bearvoyance" and item in ("weathervane", "thickclub"):
        return "bearvoyanceawakened"
    elif s == "blurrun" and item == "weathervane":
        return "blurruncharged"
    elif s == "drout" and item == "weathervane":
        return "droutdry"
    return species


def collect_pokemon_stats(battles):
    pokemon = {}
    for battle in battles.values():
        if battle.won is None:
            continue
        for mon in battle.team.values():
            name = _get_species_name(mon)
            if name not in pokemon:
                pokemon[name] = {"wins": 0, "generated": 0}
            pokemon[name]["generated"] += 1
            if battle.won:
                pokemon[name]["wins"] += 1
    return pokemon


def _pokemon_table(pokemon):
    rows = sorted(
        [
            (name, d["wins"], d["generated"], d["wins"] / d["generated"])
            for name, d in pokemon.items()
            if d["generated"] > 0
        ],
        key=lambda r: r[3],
        reverse=True,
    )
    col = max((len(r[0]) for r in rows), default=7)
    header = f"{'Pokemon':<{col}}  {'Win Rate':>8}  {'Wins':>6}  {'Generated':>9}"
    sep = "-" * len(header)
    lines = [header, sep]
    for name, wins, generated, rate in rows:
        lines.append(f"{name:<{col}}  {rate:>8.1%}  {wins:>6,}  {generated:>9,}")
    return lines


def write_winrates_train(stats):
    lines = ["Botnyak — Training Win Rates", "=" * 40]
    lines += _pokemon_table(stats.get("pokemon", {}))
    lines += ["", f"Sessions: {stats['sessions']:,}  |  Steps: {stats['total_steps']:,}"]
    open(WINRATES_TRAIN_PATH, "w").write("\n".join(lines) + "\n")


def write_winrates_serve(stats):
    lines = ["Botnyak — Serve Win Rates", "=" * 40]
    lines += _pokemon_table(stats.get("pokemon", {}))
    lines += ["", f"Sessions: {stats['sessions']:,}  |  Steps: {stats['total_steps']:,}"]
    open(WINRATES_SERVE_PATH, "w").write("\n".join(lines) + "\n")


def train():
    stats = (
        json.loads(open(STATS_PATH).read())
        if os.path.exists(STATS_PATH)
        else {"sessions": 0, "total_steps": 0, "pokemon": {}}
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

    agent = PolicyPlayer(policy=ppo.policy, battle_format=BATTLE_FORMAT, max_concurrent_battles=50)
    opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=50)
    import asyncio as _asyncio
    _asyncio.run(agent.battle_against(opponent, n_battles=100))

    for name, d in collect_pokemon_stats(agent.battles).items():
        entry = stats["pokemon"].setdefault(name, {"wins": 0, "generated": 0})
        entry["wins"] += d["wins"]
        entry["generated"] += d["generated"]

    session_wins = agent.n_won_battles
    session_wr = session_wins / 100
    ppo.save(MODEL_PATH)
    stats["sessions"] += 1
    stats["total_steps"] += 98_304
    open(STATS_PATH, "w").write(json.dumps(stats, indent=2))
    write_winrates_train(stats)
    print(f"Session {stats['sessions']} complete — {stats['total_steps']:,} steps trained | eval vs heuristics: {session_wins}/100 ({session_wr:.0%})")


async def serve():
    if not os.path.exists(MODEL_PATH + ".zip"):
        print("No trained model found. Run with --train first.")
        return

    stats = (
        json.loads(open(STATS_PATH).read())
        if os.path.exists(STATS_PATH)
        else {"sessions": 0, "total_steps": 0, "pokemon": {}}
    )

    bot = PolicyPlayer(
        policy=PPO.load(MODEL_PATH).policy,
        account_configuration=AccountConfiguration("Botnyak", os.environ.get("BOTNYAK_PASSWORD")),
        battle_format=BATTLE_FORMAT,
        avatar="ash",
        max_concurrent_battles=50,
        start_timer_on_battle_start=True,
        server_configuration=LocalhostServerConfiguration,
    )

    async def _join_lobby():
        await bot.ps_client.logged_in.wait()
        await bot.ps_client.send_message("/join lobby")

    await handle_threaded_coroutines(_join_lobby(), bot.ps_client.loop)
    print("Botnyak is online. Accepting challenges...")

    seen_battles = set()
    while True:
        await bot.accept_challenges(None, 1)
        new_battles = {tag: b for tag, b in bot.battles.items() if tag not in seen_battles}
        seen_battles.update(new_battles)
        for name, d in collect_pokemon_stats(new_battles).items():
            entry = stats["pokemon"].setdefault(name, {"wins": 0, "generated": 0})
            entry["wins"] += d["wins"]
            entry["generated"] += d["generated"]
        open(STATS_PATH, "w").write(json.dumps(stats, indent=2))
        write_winrates_serve(stats)


if __name__ == "__main__":
    import asyncio
    import sys

    if "--help" in sys.argv:
        print("""
Botnyak — a self-improving Pokémon Showdown bot.

Trains via PPO self-play in gen9swserandombattle, evaluates against
SimpleHeuristicsPlayer after each session, and accepts challenges
from users on a local Showdown server.

Usage:
  python botnyak.py                  Train one session (~98k steps)
  python botnyak.py --loops=N        Train N sessions back to back
  python botnyak.py --serve          Load saved model and accept challenges
  python botnyak.py --reset-wr       Reset pokemon win rate stats (keeps model/training progress)
  python botnyak.py --help           Show this message

Files written to botnyak/stats/:
  botnyak_model.zip       Saved PPO model (loaded automatically each session)
  botnyak_stats.json      Raw cumulative stats
  winrates_train.txt      Per-pokemon win rates from post-training eval vs SimpleHeuristicsPlayer
  winrates_serve.txt      Per-pokemon win rates from real user challenges

Config:
  botnyak/.env            Set BOTNYAK_PASSWORD=yourpassword to protect the bot's account
""".strip())
    elif "--reset-wr" in sys.argv:
        stats = (
            json.loads(open(STATS_PATH).read())
            if os.path.exists(STATS_PATH)
            else {"sessions": 0, "total_steps": 0, "pokemon": {}}
        )
        stats["pokemon"] = {}
        open(STATS_PATH, "w").write(json.dumps(stats, indent=2))
        write_winrates_train(stats)
        write_winrates_serve(stats)
        print("Win rate stats reset.")
    elif "--serve" in sys.argv:
        asyncio.run(serve())
    else:
        loops = 1
        for arg in sys.argv[1:]:
            if arg.startswith("--loops="):
                loops = int(arg.split("=")[1])
        for i in range(loops):
            if loops > 1:
                print(f"--- Loop {i + 1}/{loops} ---")
            train()
