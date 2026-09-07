"""
models.py — Модели данных ARES (Adversarial Roleplaying Equilibrium System).

Содержит Pydantic-схемы для описания боевых сущностей, навыков,
конфигурации боя и структуры отчётов СППР.
Используется Pydantic v2+ (model_dump / model_dump_json).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Типы и константы
# ---------------------------------------------------------------------------

class SkillType(str, Enum):
    """Тип навыка (определяет логику применения или визуализацию)."""
    PHYSICAL = "Физ."
    MAGICAL = "Маг."


class EffectType(str, Enum):
    """Тип статус-эффекта."""
    STUN = "Оглушение"
    DOT = "ДоТ (периодический урон)"
    SHIELD = "Щит"
    BUFF_ATTACK = "Усиление атаки"
    DEBUFF_DEFENSE = "Срез брони"


class StatusEffect(BaseModel):
    """
    Описание статус-эффекта, применяемого к бойцу.

    Attributes:
        name: Название эффекта.
        effect_type: Тип эффекта.
        value: Сила эффекта (урон ДоТ, прочность щита или дельта стата).
        duration: Длительность эффекта в ходах.
    """
    name: str = Field(..., description="Название эффекта")
    effect_type: EffectType = Field(..., description="Тип эффекта")
    value: int = Field(0, description="Сила эффекта")
    duration: int = Field(..., gt=0, description="Длительность (в ходах)")


# ---------------------------------------------------------------------------
# Навык
# ---------------------------------------------------------------------------

class Skill(BaseModel):
    """
    Описание боевого навыка.

    Attributes:
        name: Название навыка (отображается в UI).
        cost: Стоимость применения в единицах маны/энергии.
        damage: Базовый урон навыка.
        cooldown: Время перезарядки навыка в ходах (0 = нет кулдауна).
        skill_type: Тип навыка (физический / магический).
        crit_chance: Шанс критического удара (0.0-1.0).
        damage_variance: Разброс урона (например, 0.15 означает ±15%).
        applied_effect: Опциональный статус-эффект, накладываемый навыком.
        target_self: Флаг, накладывать ли эффект на себя (полезно для баффов/щитов).
    """
    name: str = Field(..., description="Название навыка")
    cost: int = Field(0, ge=0, description="Стоимость маны/энергии")
    damage: int = Field(0, ge=0, description="Базовый урон")
    cooldown: int = Field(0, ge=0, description="Кулдаун в ходах")
    skill_type: SkillType = Field(
        SkillType.PHYSICAL, description="Тип навыка (физ/маг)"
    )
    crit_chance: float = Field(0.10, ge=0.0, le=1.0, description="Шанс крита")
    damage_variance: float = Field(0.15, ge=0.0, le=1.0, description="Разброс урона")
    applied_effect: Optional[StatusEffect] = Field(None, description="Накладываемый статус-эффект")
    target_self: bool = Field(False, description="Накладывать ли эффект на себя")


# ---------------------------------------------------------------------------
# Боевая единица (Актёр)
# ---------------------------------------------------------------------------

class Actor(BaseModel):
    """
    Боевая единица — игрок или противник.

    Attributes:
        name: Имя персонажа.
        hp: Текущее здоровье.
        max_hp: Максимальное здоровье.
        mp: Текущая мана/энергия.
        max_mp: Максимальная мана/энергия.
        attack: Модификатор атаки (добавляется к базовому урону навыка).
        defense: Показатель защиты (вычитается из входящего урона).
        skills: Список доступных навыков.
        cooldowns: Текущие таймеры кулдаунов для каждого навыка.
        active_effects: Текущие статус-эффекты на бойце.
        shield: Текущая прочность активного щита.
    """
    name: str = Field(..., description="Имя персонажа")
    hp: int = Field(..., gt=0, description="Текущее HP")
    max_hp: int = Field(..., gt=0, description="Максимальное HP")
    mp: int = Field(0, ge=0, description="Текущая мана")
    max_mp: int = Field(0, ge=0, description="Максимальная мана")
    attack: int = Field(0, ge=0, description="Модификатор атаки")
    defense: int = Field(0, ge=0, description="Показатель защиты")
    skills: list[Skill] = Field(default_factory=list, description="Навыки")
    cooldowns: list[int] = Field(
        default_factory=list, description="Таймеры кулдаунов навыков"
    )
    active_effects: list[StatusEffect] = Field(
        default_factory=list, description="Активные эффекты"
    )
    shield: int = Field(0, ge=0, description="Прочность щита")

    def init_cooldowns(self) -> None:
        """Инициализирует массив кулдаунов нулями по числу навыков."""
        self.cooldowns = [0] * len(self.skills)
        self.active_effects = []
        self.shield = 0


# ---------------------------------------------------------------------------
# Конфигурация боя
# ---------------------------------------------------------------------------

class CombatConfig(BaseModel):
    """
    Глобальные правила боевого столкновения.

    Attributes:
        max_turns: Максимальное количество ходов (раундов) в бою.
        mp_regen: Количество маны, восстанавливаемой каждый ход.
        is_stochastic: Флаг стохастических механик (инициатива, криты, разброс урона).
    """
    max_turns: int = Field(50, gt=0, description="Лимит ходов на бой")
    mp_regen: int = Field(5, ge=0, description="Реген маны за ход")
    is_stochastic: bool = Field(True, description="Включен ли фактор случайности")


# ---------------------------------------------------------------------------
# Рекомендация по ребалансу
# ---------------------------------------------------------------------------

class BalanceRecommendation(BaseModel):
    """
    Одна рекомендация по корректировке параметров навыка.

    Attributes:
        skill_name: Название навыка, к которому относится рекомендация.
        reason: Обоснование рекомендации на русском языке.
        damage_delta: Предлагаемое изменение урона (может быть отрицательным).
        cooldown_delta: Предлагаемое изменение кулдауна (может быть отрицательным).
        cost_delta: Предлагаемое изменение стоимости маны.
    """
    skill_name: str = Field(..., description="Название навыка")
    reason: str = Field("", description="Обоснование")
    damage_delta: int = Field(0, description="Дельта урона")
    cooldown_delta: int = Field(0, description="Дельта кулдауна")
    cost_delta: int = Field(0, description="Дельта стоимости маны")
    effect_duration_delta: int = Field(0, description="Дельта длительности эффекта")


# ---------------------------------------------------------------------------
# Отчёт СППР (DSS)
# ---------------------------------------------------------------------------

class BalanceReport(BaseModel):
    """
    Итоговый отчёт системы поддержки принятия решений (СППР).

    Attributes:
        win_rate_vs_random: Процент побед обученного агента против случайного.
        win_rate_vs_greedy: Процент побед обученного агента против жадного.
        win_rate_vs_mirror: Процент побед обученного агента в зеркальном матче.
        avg_ttk_vs_random: Среднее количество ходов до победы (vs random).
        avg_ttk_vs_greedy: Среднее количество ходов до победы (vs greedy).
        avg_ttk_vs_mirror: Среднее количество ходов до победы (vs mirror).
        skill_usage: Словарь «навык → доля использований» (0.0–1.0).
        dominant_skills: Список навыков, признанных доминантными.
        underused_skills: Список навыков, признанных недоиспользуемыми.
        recommendations: Список рекомендаций по ребалансу.
    """
    win_rate_vs_random: float = Field(0.0, description="Win Rate vs Random (%)")
    win_rate_vs_greedy: float = Field(0.0, description="Win Rate vs Greedy (%)")
    win_rate_vs_mirror: float = Field(0.0, description="Win Rate vs Mirror (%)")
    avg_ttk_vs_random: float = Field(0.0, description="Средний TTK vs Random")
    avg_ttk_vs_greedy: float = Field(0.0, description="Средний TTK vs Greedy")
    avg_ttk_vs_mirror: float = Field(0.0, description="Средний TTK vs Mirror")
    skill_usage: dict[str, float] = Field(
        default_factory=dict, description="Частота использования навыков"
    )
    dominant_skills: list[str] = Field(
        default_factory=list, description="Доминантные навыки"
    )
    underused_skills: list[str] = Field(
        default_factory=list, description="Недоиспользуемые навыки"
    )
    recommendations: list[BalanceRecommendation] = Field(
        default_factory=list, description="Рекомендации по ребалансу"
    )


# ---------------------------------------------------------------------------
# Набор данных по умолчанию для быстрого старта
# ---------------------------------------------------------------------------

def default_skills() -> list[Skill]:
    """Возвращает стандартный набор навыков для MVP."""
    return [
        Skill(
            name="Удар мечом",
            cost=0,
            damage=25,
            cooldown=0,
            skill_type=SkillType.PHYSICAL,
            crit_chance=0.10,
            damage_variance=0.15,
        ),
        Skill(
            name="Огненный шар",
            cost=20,
            damage=35,
            cooldown=0,
            skill_type=SkillType.MAGICAL,
            crit_chance=0.0,
            damage_variance=0.10,
            applied_effect=StatusEffect(
                name="Горение",
                effect_type=EffectType.DOT,
                value=5,
                duration=2,
            ),
        ),
        Skill(
            name="Ледяная стрела",
            cost=15,
            damage=20,
            cooldown=3,
            skill_type=SkillType.MAGICAL,
            crit_chance=0.05,
            damage_variance=0.15,
            applied_effect=StatusEffect(
                name="Заморозка",
                effect_type=EffectType.STUN,
                value=0,
                duration=1,
            ),
        ),
        Skill(
            name="Каменная кожа",
            cost=25,
            damage=0,
            cooldown=3,
            skill_type=SkillType.MAGICAL,
            crit_chance=0.0,
            damage_variance=0.0,
            target_self=True,
            applied_effect=StatusEffect(
                name="Каменный щит",
                effect_type=EffectType.SHIELD,
                value=25,
                duration=2,
            ),
        ),
    ]


def default_player() -> Actor:
    """Создаёт актёра-игрока с параметрами по умолчанию."""
    player = Actor(
        name="Игрок",
        hp=120, max_hp=120,
        mp=50, max_mp=50,
        attack=8, defense=4,
        skills=default_skills(),
    )
    player.init_cooldowns()
    return player


def default_enemy() -> Actor:
    """Создаёт актёра-врага с параметрами по умолчанию."""
    enemy = Actor(
        name="Враг",
        hp=120, max_hp=120,
        mp=50, max_mp=50,
        attack=8, defense=4,
        skills=default_skills(),
    )
    enemy.init_cooldowns()
    return enemy


def default_config() -> CombatConfig:
    """Возвращает конфигурацию боя по умолчанию."""
    return CombatConfig(max_turns=50, mp_regen=5)
