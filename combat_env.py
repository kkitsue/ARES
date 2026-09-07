"""
combat_env.py — Кастомная среда Gymnasium с поддержкой Action Masking для ARES.

Оборачивает CombatSimulator в gym.Env интерфейс:
  - Нормализованный вектор наблюдений (Box).
  - Дискретное пространство действий (навыки + пропуск хода).
  - Метод action_masks() для интеграции с MaskablePPO через ActionMasker.
  - Полный раунд за один step: ход игрока → проверка → ход врага → проверка.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models import Actor, CombatConfig, default_config, default_enemy, default_player
from simulator import CombatSimulator


class CombatEnv(gym.Env):
    """
    Gymnasium-среда для пошагового 1v1 RPG-боя.

    Один вызов step() = полный раунд:
      1. Агент (игрок) применяет выбранное действие.
      2. Проверка: если враг повержен → +1.0, terminated.
      3. Если враг жив — враг делает ход (в зависимости от enemy_type).
      4. Проверка: если игрок повержен → -1.0, terminated.
      5. Если оба живы — штраф -0.01 за затягивание.
      6. Обновление кулдаунов и регенерация маны обоих бойцов.

    Метод action_masks() возвращает булев массив допустимых действий,
    который затем используется обёрткой ActionMasker из sb3-contrib.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        player: Actor | None = None,
        enemy: Actor | None = None,
        config: CombatConfig | None = None,
        render_mode: str | None = None,
        enemy_type: str = "random",
        enemy_model: Any | None = None,
    ) -> None:
        """
        Инициализация среды.

        Args:
            player: Шаблон актёра-игрока (если None — используется default).
            enemy: Шаблон актёра-врага (если None — используется default).
            config: Конфигурация боя.
            render_mode: Режим отрисовки (пока не используется).
            enemy_type: Тип поведения врага ("random", "greedy", "mirror").
            enemy_model: Обученная модель для режима "mirror".
        """
        super().__init__()

        self.render_mode = render_mode
        self.enemy_type = enemy_type
        self.enemy_model = enemy_model

        # Инициализируем шаблоны бойцов
        self._player_template = player or default_player()
        self._enemy_template = enemy or default_enemy()
        self._config = config or default_config()

        # Создаём симулятор
        self.sim = CombatSimulator(
            player=self._player_template,
            enemy=self._enemy_template,
            config=self._config,
        )

        # Количество навыков (одинаковое у игрока, по шаблону)
        self.num_skills = len(self._player_template.skills)

        # Размер вектора наблюдений:
        #   4 (HP/MP) + num_skills (кулдауны) + 6 (статусы)
        self.obs_size = 4 + self.num_skills + 6

        # Пространство наблюдений — нормализованные значения [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_size,),
            dtype=np.float32,
        )

        # Пространство действий: num_skills навыков + 1 пропуск хода
        self.action_space = spaces.Discrete(self.num_skills + 1)

    # ------------------------------------------------------------------
    # Формирование вектора наблюдений
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Формирует нормализованный вектор наблюдений из текущего состояния боя (для игрока).
        """
        state = self.sim.get_state()
        obs = [
            state["player_hp"],
            state["player_mp"],
            state["enemy_hp"],
            state["enemy_mp"],
        ]
        obs.extend(state["player_cooldowns"])
        obs.extend([
            state["player_stunned"],
            state["player_shield"],
            state["player_dot_turns"],
            state["enemy_stunned"],
            state["enemy_shield"],
            state["enemy_dot_turns"],
        ])
        return np.array(obs, dtype=np.float32)

    def _get_enemy_obs(self) -> np.ndarray:
        """
        Формирует нормализованный вектор наблюдений от лица врага (для Mirror режима).
        """
        state = self.sim.get_state()
        obs = [
            state["enemy_hp"],
            state["enemy_mp"],
            state["player_hp"],
            state["player_mp"],
        ]
        obs.extend(state["enemy_cooldowns"])
        obs.extend([
            state["enemy_stunned"],
            state["enemy_shield"],
            state["enemy_dot_turns"],
            state["player_stunned"],
            state["player_shield"],
            state["player_dot_turns"],
        ])
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Маска действий для MaskablePPO
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Возвращает булев массив допустимых действий для текущего игрока.
        """
        mask = self.sim.get_valid_actions(self.sim.player)
        return np.array(mask, dtype=bool)

    # ------------------------------------------------------------------
    # Ход противника
    # ------------------------------------------------------------------

    def _enemy_turn(self) -> tuple[float, int, bool]:
        """
        Логика выбора и выполнения действия врагом.
        Returns:
            (урон, индекс действия, был ли оглушен)
        """
        is_stunned = self.sim.start_turn_phase(self.sim.enemy)
        
        if is_stunned:
            enemy_action = self.num_skills
        else:
            enemy_mask = self.sim.get_valid_actions(self.sim.enemy)
            valid_indices = [i for i, v in enumerate(enemy_mask) if v]

            if self.enemy_type == "greedy":
                best_action = len(enemy_mask) - 1
                best_damage = -1
                for i, is_valid in enumerate(enemy_mask):
                    if is_valid and i < self.num_skills:
                        if self.sim.enemy.skills[i].damage > best_damage:
                            best_damage = self.sim.enemy.skills[i].damage
                            best_action = i
                enemy_action = best_action
            elif self.enemy_type == "mirror" and self.enemy_model is not None:
                enemy_obs = self._get_enemy_obs()
                enemy_action, _ = self.enemy_model.predict(
                    enemy_obs,
                    deterministic=True,
                    action_masks=np.array(enemy_mask, dtype=bool)
                )
            else: # "random"
                enemy_action = self.np_random.choice(valid_indices)

        # Выполняем действие только если враг жив (мог умереть от DoT)
        if self.sim.enemy.hp > 0:
            enemy_damage = self.sim.execute_action(
                self.sim.enemy, self.sim.player, int(enemy_action)
            )
        else:
            enemy_damage = 0
            
        return enemy_damage, int(enemy_action), is_stunned

    # ------------------------------------------------------------------
    # Сброс среды
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Сбрасывает среду в начальное состояние.
        Если включен стохастизм, бросается инициатива.
        """
        super().reset(seed=seed, options=options)
        self.sim.rng.seed(self.np_random.integers(0, 2**32 - 1))
        self.sim.reset()
        
        info = {}
        # Бросок инициативы, если включены стохастические механики
        if self._config.is_stochastic:
            if self.np_random.random() < 0.5:
                # Враг ходит первым
                dmg, act, stunned = self._enemy_turn()
                info["initiative"] = "enemy"
                info["first_enemy_action"] = act
                info["first_enemy_damage"] = dmg
                
                # Завершаем ход врага (тики баффов/дебаффов, кулдауны, мана)
                self.sim.end_turn_phase(self.sim.enemy)
                self.sim.tick_cooldowns(self.sim.enemy)
                self.sim.regen_mp(self.sim.enemy)
            else:
                info["initiative"] = "player"
        else:
            info["initiative"] = "player"

        return self._get_obs(), info

    # ------------------------------------------------------------------
    # Шаг среды (полный раунд)
    # ------------------------------------------------------------------

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Выполняет полный раунд боя.
        """
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        # --- 1. Ход игрока ---
        p_stunned = self.sim.start_turn_phase(self.sim.player)
        info["player_stunned_this_turn"] = p_stunned
        
        if self.sim.player.hp <= 0:
            # Умер от DoT
            reward = -1.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, info
            
        if p_stunned:
            action = self.num_skills
            
        player_damage = self.sim.execute_action(
            self.sim.player, self.sim.enemy, action
        )
        info["player_action"] = action
        info["player_damage"] = player_damage

        # --- 2. Проверка: враг повержен? ---
        if self.sim.enemy.hp <= 0:
            reward = 1.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, info

        # --- 3. Ход врага ---
        enemy_damage, enemy_action, e_stunned = self._enemy_turn()
        info["enemy_action"] = enemy_action
        info["enemy_damage"] = enemy_damage
        info["enemy_stunned_this_turn"] = e_stunned

        # --- 4. Проверка: игрок повержен? ---
        if self.sim.player.hp <= 0:
            reward = -1.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, info

        # --- 5. Оба живы — штраф за затягивание ---
        reward = -0.01

        # --- 6. Конец раунда (эффекты, кулдауны, мана) ---
        self.sim.end_turn_phase(self.sim.player)
        self.sim.end_turn_phase(self.sim.enemy)
        
        self.sim.tick_cooldowns(self.sim.player)
        self.sim.tick_cooldowns(self.sim.enemy)
        self.sim.regen_mp(self.sim.player)
        self.sim.regen_mp(self.sim.enemy)

        # Увеличиваем счётчик ходов
        self.sim.turn += 1

        # Проверка лимита ходов
        if self.sim.turn >= self.sim.config.max_turns:
            truncated = True
            reward = -0.5

        return self._get_obs(), reward, terminated, truncated, info
