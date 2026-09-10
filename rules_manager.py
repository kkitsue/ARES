"""
rules_manager.py — Модуль управления JSON-пресетами правил ARES.

Обеспечивает:
  - Модель RulesPreset (Pydantic v2): контейнер для CombatConfig,
    Actor (player/enemy) и списка Skill.
  - Функции load_rules / save_rules для сериализации пресетов.
  - Генерацию и сохранение файла default_rules.json при первом запуске.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from models import (
    Actor,
    BalanceReport,
    CombatConfig,
    Skill,
    default_config,
    default_enemy,
    default_player,
    default_skills,
)


# Путь к файлу правил по умолчанию рядом с main.py
DEFAULT_RULES_PATH = Path(__file__).parent / "default_rules.json"


# ---------------------------------------------------------------------------
# Контейнер пресета правил
# ---------------------------------------------------------------------------

class RulesPreset(BaseModel):
    """
    Полный пресет правил боевого столкновения.

    Объединяет конфигурацию боя, параметры игрока и врага,
    а также список навыков, общих для обоих бойцов.

    Attributes:
        name: Название пресета (например, "Default RPG Settings").
        combat_config: Глобальные настройки боя (ходы, реген, стохастика).
        player: Характеристики и начальные навыки игрока.
        enemy: Характеристики и начальные навыки противника (по умолчанию).
        skills: Полный реестр умений, доступных в игре (словарь/список).
        enemies_pool: Опциональный пул противников для случайного выбора.
    """

    name: str = Field("Default Rules", description="Название пресета")
    combat_config: CombatConfig = Field(default_factory=default_config)
    player: Actor = Field(default_factory=default_player)
    enemy: Actor = Field(default_factory=default_enemy)
    skills: list[Skill] = Field(default_factory=default_skills)
    enemies_pool: list[Actor] = Field(default_factory=list, description="Пул возможных противников")

    def build_player(self) -> Actor:
        """Возвращает нового актёра-игрока с навыками из пресета."""
        p = self.player.model_copy(deep=True)
        p.skills = [s.model_copy(deep=True) for s in self.skills]
        p.init_cooldowns()
        return p

    def build_enemy(self) -> Actor:
        """Возвращает нового актёра-врага с навыками из пресета."""
        e = self.enemy.model_copy(deep=True)
        e.skills = [s.model_copy(deep=True) for s in self.skills]
        e.init_cooldowns()
        return e


# ---------------------------------------------------------------------------
# Сериализация
# ---------------------------------------------------------------------------

def save_rules(preset: RulesPreset, path: str | Path) -> None:
    """
    Сохраняет пресет правил в JSON-файл.

    Args:
        preset: Объект RulesPreset для сохранения.
        path: Путь к выходному файлу.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(preset.model_dump_json(indent=2))
        f.write("\n")


def load_rules(path: str | Path) -> RulesPreset:
    """
    Загружает пресет правил из JSON-файла.

    Если файл не существует, генерирует дефолтный пресет и сохраняет его.

    Args:
        path: Путь к файлу пресета.

    Returns:
        Загруженный или сгенерированный объект RulesPreset.
    """
    path = Path(path)
    if not path.exists():
        preset = build_default_preset()
        save_rules(preset, path)
        return preset

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return RulesPreset.model_validate(data)


def build_default_preset() -> RulesPreset:
    """
    Создаёт дефолтный пресет правил из моделей по умолчанию.

    Returns:
        Объект RulesPreset с заводскими настройками ARES.
    """
    skills = default_skills()

    player = default_player()
    player.skills = skills
    player.init_cooldowns()

    enemy = default_enemy()
    enemy.skills = skills
    enemy.init_cooldowns()

    return RulesPreset(
        name="default",
        combat_config=default_config(),
        player=player,
        enemy=enemy,
        skills=skills,
    )


def ensure_default_rules() -> RulesPreset:
    """
    Гарантирует наличие файла default_rules.json.

    Если файл уже существует - загружает его.
    Если нет - создаёт дефолтный пресет и сохраняет.

    Returns:
        Активный пресет правил.
    """
    return load_rules(DEFAULT_RULES_PATH)

# ---------------------------------------------------------------------------
# Авто-ребаланс
# ---------------------------------------------------------------------------

def apply_balance_recommendations(
    preset: RulesPreset,
    report: BalanceReport,
    apply_frozen_buffs: bool = False
) -> tuple[RulesPreset, list[str]]:
    """
    Применяет рекомендации из отчёта СППР к текущему пресету.

    Args:
        preset: Текущий пресет правил.
        report: Отчёт с рекомендациями.
        apply_frozen_buffs: Если False, применяет только нерфы (пропускает
                            баффы со статусом заморозки).

    Returns:
        tuple(обновлённый клон RulesPreset, список строк-диффов изменений).
    """
    new_preset = preset.model_copy(deep=True)
    diff_log = []

    for rec in report.recommendations:
        # Проверяем, нужно ли применять эту рекомендацию
        if not apply_frozen_buffs and "ЗАМОРОЖЕНО" in rec.reason:
            continue

        # Ищем навык в пресете
        skill_idx = next((i for i, s in enumerate(new_preset.skills) if s.name == rec.skill_name), None)
        if skill_idx is None:
            continue

        skill = new_preset.skills[skill_idx]

        # Сохраняем старые значения для диффа
        old_dmg = skill.damage
        old_cd = skill.cooldown
        old_cost = skill.cost

        # Применяем дельты
        skill.damage = max(0, skill.damage + rec.damage_delta)
        skill.cooldown = max(0, skill.cooldown + rec.cooldown_delta)
        skill.cost = max(0, skill.cost + rec.cost_delta)

        # Формируем строку диффа
        diff_str = f"[bold cyan]{skill.name}[/bold cyan]: "
        changes = []
        if old_dmg != skill.damage:
            changes.append(f"Урон ({old_dmg} -> {skill.damage})")
        if old_cd != skill.cooldown:
            changes.append(f"Кулдаун ({old_cd} -> {skill.cooldown})")
        if old_cost != skill.cost:
            changes.append(f"Стоимость ({old_cost} -> {skill.cost})")

        if changes:
            diff_str += ", ".join(changes)
            diff_log.append(diff_str)

    return new_preset, diff_log
