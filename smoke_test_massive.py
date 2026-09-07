"""
smoke_test_massive.py — Быстрый smoke-тест обучения MaskablePPO на massive_rules.json.

Проверяет:
  1. Корректную загрузку пресета (50 навыков).
  2. Правильность пространств (Action: Discrete(51), Obs: Box(60)).
  3. Успешное выполнение 100 шагов обучения без ошибок размерностей.
  4. Успешный прогон 20 эпизодов оценки.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from rules_manager import load_rules
from combat_env import CombatEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

RULES_PATH = Path(__file__).parent / "massive_rules.json"


def mask_fn(env: CombatEnv) -> np.ndarray:
    """Функция маскирования для ActionMasker."""
    return env.action_masks()


def main() -> None:
    print("[INFO] Загрузка massive_rules.json...")
    preset = load_rules(RULES_PATH)
    n_skills = len(preset.skills)
    print(f"[OK]   Пресет: '{preset.name}' | навыков: {n_skills}")

    player = preset.build_player()
    enemy = preset.build_enemy()
    config = preset.combat_config

    # Инициализация среды
    env_raw = CombatEnv(player=player, enemy=enemy, config=config, enemy_type="random")
    env = ActionMasker(env_raw, mask_fn)

    # Проверка пространств
    act_n = env_raw.action_space.n
    obs_n = env_raw.observation_space.shape[0]
    assert act_n == n_skills + 1, f"Action space mismatch: {act_n} != {n_skills + 1}"
    assert obs_n == 4 + n_skills + 6, f"Obs space mismatch: {obs_n} != {4 + n_skills + 6}"
    print(f"[OK]   Action Space: Discrete({act_n})")
    print(f"[OK]   Obs Space:    Box({obs_n})")

    # Smoke-test: 100 шагов обучения
    print("[INFO] Запуск обучения MaskablePPO (100 шагов)...")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
    )
    model.learn(total_timesteps=100)
    print("[OK]   100 шагов обучения выполнены без ошибок")

    # Smoke-test: 20 эпизодов оценки
    print("[INFO] Запуск оценки (20 эпизодов)...")
    wins = 0
    for ep in range(20):
        obs, _ = env_raw.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            masks = env_raw.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=False)
            obs, reward, terminated, truncated, _ = env_raw.step(int(action))
            if terminated and reward > 0:
                wins += 1

    print(f"[OK]   Оценка завершена: {wins}/20 побед")
    print("[OK]   Smoke-тест PASSED — система корректно работает с 50 навыками")


if __name__ == "__main__":
    main()
