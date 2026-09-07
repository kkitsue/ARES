"""
simulator.py — Детерминированный пошаговый боевой движок 1v1 для ARES.

Обрабатывает кулдауны, затраты маны, расчёт урона с учётом защиты,
регенерацию маны и условия завершения боя.
"""

from __future__ import annotations

import copy
import math
import numpy as np
from typing import Optional

from models import Actor, CombatConfig, Skill


class CombatSimulator:
    """
    Пошаговый боевой движок для 1v1 столкновения.

    Управляет двумя бойцами (player и enemy), отслеживает кулдауны,
    ману и номер текущего хода. Предоставляет интерфейс для среды
    Gymnasium: маска действий, применение действия, получение состояния.
    Поддерживает стохастические механики (разброс урона, криты).
    """

    def __init__(
        self,
        player: Actor,
        enemy: Actor,
        config: CombatConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Инициализация симулятора.

        Args:
            player: Шаблон актёра-игрока (будет скопирован при reset).
            enemy: Шаблон актёра-врага (будет скопирован при reset).
            config: Конфигурация боя (лимит ходов, реген маны, стохастизм).
            seed: Зерно генератора случайных чисел для воспроизводимости.
        """
        # Сохраняем шаблоны для сброса в начальное состояние
        self._player_template = player
        self._enemy_template = enemy
        self.config = config or CombatConfig()
        self.rng = np.random.RandomState(seed)

        # Текущее состояние (заполняется при reset)
        self.player: Actor = None  # type: ignore[assignment]
        self.enemy: Actor = None   # type: ignore[assignment]
        self.turn: int = 0

        # Первичный сброс
        self.reset()

    # ------------------------------------------------------------------
    # Сброс состояния
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Полный сброс боя в начальное состояние.

        Создаёт глубокие копии шаблонов бойцов, обнуляет кулдауны
        и счётчик ходов.
        """
        self.player = copy.deepcopy(self._player_template)
        self.enemy = copy.deepcopy(self._enemy_template)

        # Гарантируем инициализацию кулдаунов
        self.player.init_cooldowns()
        self.enemy.init_cooldowns()

        self.turn = 0

    # ------------------------------------------------------------------
    # Фаза начала хода (обработка DoT и STUN)
    # ------------------------------------------------------------------

    def start_turn_phase(self, actor: Actor) -> bool:
        """
        Фаза начала хода: обрабатывает периодический урон (DoT) и проверяет оглушение.
        
        Args:
            actor: Боец, чей ход начинается.
            
        Returns:
            bool: True, если боец оглушен (должен пропустить ход).
        """
        is_stunned = False
        for effect in actor.active_effects:
            if effect.effect_type == "ДоТ (периодический урон)": # EffectType.DOT
                actor.hp = max(0, actor.hp - effect.value)
            elif effect.effect_type == "Оглушение": # EffectType.STUN
                is_stunned = True
                
        return is_stunned

    # ------------------------------------------------------------------
    # Маска действий
    # ------------------------------------------------------------------

    def get_valid_actions(self, actor: Actor) -> list[bool]:
        """
        Возвращает булев список валидных действий для указанного актёра.

        Если на бойце висит STUN, все навыки недоступны.
        В противном случае навык доступен, если хватает маны и нет кулдауна.
        Последний элемент списка — пропуск хода (всегда доступен).
        """
        mask: list[bool] = []
        is_stunned = any(e.effect_type == "Оглушение" for e in actor.active_effects)
        
        for i, skill in enumerate(actor.skills):
            if is_stunned:
                mask.append(False)
            else:
                can_use = (actor.mp >= skill.cost) and (actor.cooldowns[i] == 0)
                mask.append(can_use)
                
        # Пропуск хода — всегда доступен
        mask.append(True)
        return mask

    # ------------------------------------------------------------------
    # Выполнение действия
    # ------------------------------------------------------------------

    def execute_action(
        self,
        actor: Actor,
        target: Actor,
        action_idx: int,
    ) -> int:
        """
        Применяет действие актёра к цели.
        """
        num_skills = len(actor.skills)

        # Пропуск хода — ничего не происходит
        if action_idx >= num_skills:
            return 0

        skill = actor.skills[action_idx]

        # Списываем ману
        actor.mp = max(0, actor.mp - skill.cost)

        # Учитываем баффы и дебаффы
        atk_buff = sum(e.value for e in actor.active_effects if e.effect_type == "Усиление атаки")
        def_debuff = sum(e.value for e in target.active_effects if e.effect_type == "Срез брони")
        
        effective_attack = actor.attack + atk_buff
        effective_defense = max(0, target.defense - def_debuff)

        # Рассчитываем базовый урон
        raw_damage = skill.damage + effective_attack - effective_defense
        
        if self.config.is_stochastic and skill.damage > 0:
            variance_min = 1.0 - skill.damage_variance
            variance_max = 1.0 + skill.damage_variance
            variance = self.rng.uniform(variance_min, variance_max)
            raw_damage = raw_damage * variance
            
            if self.rng.rand() < skill.crit_chance:
                raw_damage = raw_damage * 1.5

        final_damage = max(1, math.ceil(raw_damage)) if skill.damage > 0 else 0

        # Поглощение щитом
        if final_damage > 0:
            if target.shield > 0:
                if target.shield >= final_damage:
                    target.shield -= final_damage
                    final_damage = 0
                else:
                    final_damage -= target.shield
                    target.shield = 0
            
            target.hp = max(0, target.hp - final_damage)

        # Наложение эффекта
        if skill.applied_effect is not None:
            effect_target = actor if skill.target_self else target
            # Ищем существующий эффект с таким же именем
            existing = next((e for e in effect_target.active_effects if e.name == skill.applied_effect.name), None)
            if existing:
                # Обновляем длительность (refresh)
                existing.duration = max(existing.duration, skill.applied_effect.duration)
            else:
                # Добавляем новый эффект
                effect_copy = copy.deepcopy(skill.applied_effect)
                effect_target.active_effects.append(effect_copy)
                
            # Если это щит, обновляем прочность
            if skill.applied_effect.effect_type == "Щит":
                effect_target.shield = skill.applied_effect.value

        # Устанавливаем кулдаун навыка
        if skill.cooldown > 0:
            actor.cooldowns[action_idx] = skill.cooldown

        return final_damage

    # ------------------------------------------------------------------
    # Обновление кулдаунов
    # ------------------------------------------------------------------

    def tick_cooldowns(self, actor: Actor) -> None:
        """
        Уменьшает все активные кулдауны актёра на 1.

        Args:
            actor: Боевая единица для обновления таймеров.
        """
        for i in range(len(actor.cooldowns)):
            if actor.cooldowns[i] > 0:
                actor.cooldowns[i] -= 1

    # ------------------------------------------------------------------
    # Регенерация маны
    # ------------------------------------------------------------------

    def regen_mp(self, actor: Actor) -> None:
        """
        Восстанавливает ману актёра на величину mp_regen из конфигурации.

        Мана не может превысить max_mp.

        Args:
            actor: Боевая единица для восстановления маны.
        """
        actor.mp = min(actor.max_mp, actor.mp + self.config.mp_regen)

    # ------------------------------------------------------------------
    # Фаза конца хода
    # ------------------------------------------------------------------

    def end_turn_phase(self, actor: Actor) -> None:
        """
        Фаза конца хода: декремент длительностей эффектов, удаление истекших,
        обновление щита.
        """
        active = []
        for effect in actor.active_effects:
            effect.duration -= 1
            if effect.duration > 0:
                active.append(effect)
            else:
                # Если щит спадает, обнуляем его прочность
                if effect.effect_type == "Щит":
                    actor.shield = 0
        actor.active_effects = active

    # ------------------------------------------------------------------
    # Проверка завершения боя
    # ------------------------------------------------------------------

    def is_battle_over(self) -> bool:
        """
        Проверяет, завершился ли бой.

        Бой заканчивается, если:
          - HP игрока <= 0 (поражение).
          - HP врага <= 0 (победа).
          - Номер хода >= max_turns (ничья / таймаут).

        Returns:
            True, если бой завершён.
        """
        if self.player.hp <= 0:
            return True
        if self.enemy.hp <= 0:
            return True
        if self.turn >= self.config.max_turns:
            return True
        return False

    def get_winner(self) -> Optional[str]:
        """
        Определяет победителя после завершения боя.

        Returns:
            "player" — если враг повержен,
            "enemy"  — если игрок повержен,
            None     — если бой завершился по таймауту или ещё идёт.
        """
        if self.enemy.hp <= 0:
            return "player"
        if self.player.hp <= 0:
            return "enemy"
        return None

    # ------------------------------------------------------------------
    # Вектор состояния (для среды Gymnasium)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """
        Возвращает словарь с текущим состоянием боя для формирования
        вектора наблюдений в среде Gymnasium.

        Все значения нормализованы в диапазон [0, 1].
        """
        # Нормализация HP, MP
        p_hp = self.player.hp / max(1, self.player.max_hp)
        p_mp = self.player.mp / max(1, self.player.max_mp)
        e_hp = self.enemy.hp / max(1, self.enemy.max_hp)
        e_mp = self.enemy.mp / max(1, self.enemy.max_mp)

        # Нормализация кулдаунов игрока
        player_cds: list[float] = []
        for i, skill in enumerate(self.player.skills):
            max_cd = max(1, skill.cooldown)
            current_cd = self.player.cooldowns[i] / max_cd if skill.cooldown > 0 else 0.0
            player_cds.append(current_cd)

        # Нормализация кулдаунов врага
        enemy_cds: list[float] = []
        for i, skill in enumerate(self.enemy.skills):
            max_cd = max(1, skill.cooldown)
            current_cd = self.enemy.cooldowns[i] / max_cd if skill.cooldown > 0 else 0.0
            enemy_cds.append(current_cd)
            
        # Нормализация статусов игрока
        p_stunned = 1.0 if any(e.effect_type == "Оглушение" for e in self.player.active_effects) else 0.0
        p_shield = min(self.player.shield / max(1, self.player.max_hp), 1.0)
        p_dot_turns = 0.0
        for e in self.player.active_effects:
            if e.effect_type == "ДоТ (периодический урон)":
                p_dot_turns = min(e.duration / 10.0, 1.0)
                break
                
        # Нормализация статусов врага
        e_stunned = 1.0 if any(e.effect_type == "Оглушение" for e in self.enemy.active_effects) else 0.0
        e_shield = min(self.enemy.shield / max(1, self.enemy.max_hp), 1.0)
        e_dot_turns = 0.0
        for e in self.enemy.active_effects:
            if e.effect_type == "ДоТ (периодический урон)":
                e_dot_turns = min(e.duration / 10.0, 1.0)
                break

        return {
            "player_hp": p_hp,
            "player_mp": p_mp,
            "enemy_hp": e_hp,
            "enemy_mp": e_mp,
            "player_cooldowns": player_cds,
            "enemy_cooldowns": enemy_cds,
            "player_stunned": p_stunned,
            "player_shield": p_shield,
            "player_dot_turns": p_dot_turns,
            "enemy_stunned": e_stunned,
            "enemy_shield": e_shield,
            "enemy_dot_turns": e_dot_turns,
        }
