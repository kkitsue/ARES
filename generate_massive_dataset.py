"""
generate_massive_dataset.py — Генератор массивного датасета правил ARES.

Создаёт massive_rules.json с пулом из 50 уникальных тематически
сгруппированных навыков, увеличенными статами бойцов и конфигом
для масштабного стресс-тестирования.

Навыки организованы по 6 школам:
  1. Воинское мастерство  (10 навыков)
  2. Пиромантия            (8 навыков)
  3. Криомантия            (8 навыков)
  4. Тени и Яды            (8 навыков)
  5. Свет и Защита         (8 навыков)
  6. Запретная магия       (8 навыков)

После генерации выполняет Pydantic-валидацию и сохраняет файл.
"""

from __future__ import annotations

from pathlib import Path

from models import (
    Actor,
    CombatConfig,
    EffectType,
    Skill,
    SkillType,
    StatusEffect,
)
from rules_manager import RulesPreset, save_rules

OUTPUT_PATH = Path(__file__).parent / "massive_rules.json"


# ---------------------------------------------------------------------------
# Вспомогательные фабрики
# ---------------------------------------------------------------------------

def dot(name: str, dmg: int, turns: int) -> StatusEffect:
    """Фабрика StatusEffect типа DOT."""
    return StatusEffect(name=name, effect_type=EffectType.DOT, value=dmg, duration=turns)


def stun(name: str, turns: int = 1) -> StatusEffect:
    """Фабрика StatusEffect типа STUN."""
    return StatusEffect(name=name, effect_type=EffectType.STUN, value=0, duration=turns)


def shield(name: str, val: int, turns: int) -> StatusEffect:
    """Фабрика StatusEffect типа SHIELD."""
    return StatusEffect(name=name, effect_type=EffectType.SHIELD, value=val, duration=turns)


def buff_atk(name: str, val: int, turns: int) -> StatusEffect:
    """Фабрика StatusEffect типа BUFF_ATTACK."""
    return StatusEffect(name=name, effect_type=EffectType.BUFF_ATTACK, value=val, duration=turns)


def debuff_def(name: str, val: int, turns: int) -> StatusEffect:
    """Фабрика StatusEffect типа DEBUFF_DEFENSE."""
    return StatusEffect(name=name, effect_type=EffectType.DEBUFF_DEFENSE, value=val, duration=turns)


def phys(
    name: str,
    cost: int,
    dmg: int,
    cd: int,
    effect: StatusEffect | None = None,
    target_self: bool = False,
    crit: float = 0.10,
    variance: float = 0.15,
) -> Skill:
    """Фабрика физического навыка."""
    return Skill(
        name=name,
        cost=cost,
        damage=dmg,
        cooldown=cd,
        skill_type=SkillType.PHYSICAL,
        crit_chance=crit,
        damage_variance=variance,
        applied_effect=effect,
        target_self=target_self,
    )


def mag(
    name: str,
    cost: int,
    dmg: int,
    cd: int,
    effect: StatusEffect | None = None,
    target_self: bool = False,
    crit: float = 0.05,
    variance: float = 0.10,
) -> Skill:
    """Фабрика магического навыка."""
    return Skill(
        name=name,
        cost=cost,
        damage=dmg,
        cooldown=cd,
        skill_type=SkillType.MAGICAL,
        crit_chance=crit,
        damage_variance=variance,
        applied_effect=effect,
        target_self=target_self,
    )


# ---------------------------------------------------------------------------
# Школа 1: Воинское мастерство (10 навыков)
# ---------------------------------------------------------------------------
WARRIOR_SKILLS: list[Skill] = [
    phys("Быстрый выпад",     cost=0,  dmg=20, cd=0),
    phys("Рубящий удар",      cost=5,  dmg=28, cd=1),
    phys("Двойной удар",      cost=10, dmg=36, cd=2, crit=0.15),
    phys("Сокрушающий удар",  cost=15, dmg=45, cd=2, crit=0.12, variance=0.20),
    phys("Раскол брони",      cost=10, dmg=22, cd=3, effect=debuff_def("Раскол", 6, 2)),
    phys("Оглушающий удар",   cost=15, dmg=20, cd=3, effect=stun("Контузия")),
    phys("Свирепый натиск",   cost=20, dmg=40, cd=3, crit=0.20, variance=0.25),
    phys("Вихрь клинков",     cost=20, dmg=35, cd=4, crit=0.15),
    phys("Казнь",             cost=25, dmg=60, cd=4, crit=0.25, variance=0.30),
    phys("Геройский удар",    cost=30, dmg=75, cd=5, crit=0.30, variance=0.20),
]

# ---------------------------------------------------------------------------
# Школа 2: Пиромантия (8 навыков)
# ---------------------------------------------------------------------------
PYRO_SKILLS: list[Skill] = [
    mag("Искра",             cost=5,  dmg=18, cd=0),
    mag("Огненный заряд",   cost=10, dmg=28, cd=1),
    mag("Огненный шар",     cost=20, dmg=38, cd=0, effect=dot("Горение-I", 5, 2)),
    mag("Пылающая стрела",  cost=15, dmg=30, cd=2, effect=dot("Горение-II", 7, 2)),
    mag("Огненная буря",    cost=35, dmg=50, cd=3, effect=dot("Пожар", 10, 2)),
    mag("Пиробомба",        cost=25, dmg=45, cd=3, crit=0.10, variance=0.20),
    mag("Жертвенный огонь", cost=40, dmg=65, cd=4, effect=dot("Горение-III", 12, 3)),
    mag("Метеор",           cost=55, dmg=90, cd=5, variance=0.25),
]

# ---------------------------------------------------------------------------
# Школа 3: Криомантия (8 навыков)
# ---------------------------------------------------------------------------
CRYO_SKILLS: list[Skill] = [
    mag("Ледяная игла",     cost=5,  dmg=16, cd=0),
    mag("Ледяная стрела",   cost=15, dmg=28, cd=2, effect=stun("Заморозка")),
    mag("Ледяная цепь",     cost=20, dmg=25, cd=3, effect=stun("Оцепенение")),
    mag("Ледяные оковы",    cost=25, dmg=30, cd=4, effect=stun("Ледяной захват")),
    mag("Ледяной щит",      cost=25, dmg=0,  cd=3, effect=shield("Ледяной барьер", 35, 2), target_self=True),
    mag("Ледяная броня",    cost=20, dmg=0,  cd=4, effect=shield("Крио-броня", 50, 3), target_self=True),
    mag("Ледяное копьё",    cost=30, dmg=50, cd=3, variance=0.15),
    mag("Буран",            cost=45, dmg=70, cd=5, effect=dot("Обморожение", 8, 2)),
]

# ---------------------------------------------------------------------------
# Школа 4: Тени и Яды (8 навыков)
# ---------------------------------------------------------------------------
SHADOW_SKILLS: list[Skill] = [
    phys("Ядовитый укол",   cost=5,  dmg=12, cd=1, effect=dot("Яд-I", 5, 3)),
    phys("Ядовитый клинок", cost=10, dmg=18, cd=1, effect=dot("Яд-II", 7, 3)),
    phys("Рассечение",      cost=15, dmg=30, cd=2, effect=dot("Кровотечение", 6, 3)),
    phys("Смертельный яд",  cost=20, dmg=22, cd=3, effect=dot("Яд-III", 10, 3)),
    phys("Ночная охота",    cost=15, dmg=38, cd=2, crit=0.25, variance=0.20),
    phys("Теневой рывок",   cost=20, dmg=45, cd=3, crit=0.20),
    phys("Боевой клич",     cost=20, dmg=0,  cd=3, effect=buff_atk("Берсерк", 8, 2), target_self=True),
    phys("Смертельный удар", cost=30, dmg=55, cd=4, crit=0.30, variance=0.25),
]

# ---------------------------------------------------------------------------
# Школа 5: Свет и Защита (8 навыков)
# ---------------------------------------------------------------------------
LIGHT_SKILLS: list[Skill] = [
    mag("Магическая искра",   cost=5,  dmg=16, cd=0),
    mag("Луч света",          cost=10, dmg=24, cd=1),
    mag("Астральный барьер",  cost=25, dmg=0,  cd=3, effect=shield("Астральный щит", 40, 2), target_self=True),
    mag("Каменная стойка",    cost=15, dmg=0,  cd=3, effect=shield("Земляной щит", 25, 3), target_self=True),
    mag("Благословение",      cost=20, dmg=0,  cd=3, effect=buff_atk("Благодать", 6, 2), target_self=True),
    mag("Священный щит",      cost=35, dmg=0,  cd=4, effect=shield("Сакральная броня", 60, 2), target_self=True),
    mag("Возмездие",          cost=30, dmg=55, cd=4),
    mag("Свет Справедливости", cost=40, dmg=70, cd=5),
]

# ---------------------------------------------------------------------------
# Школа 6: Запретная магия / Хаос (8 навыков)
# ---------------------------------------------------------------------------
CHAOS_SKILLS: list[Skill] = [
    mag("Тёмный импульс",   cost=20, dmg=40, cd=2, effect=debuff_def("Уязвимость", 4, 2)),
    mag("Хаотический взрыв", cost=30, dmg=55, cd=3, variance=0.30),
    mag("Поглощение душ",   cost=25, dmg=35, cd=3, effect=dot("Тёмный яд", 12, 3)),
    mag("Разрыв реальности", cost=40, dmg=70, cd=4, variance=0.35),
    mag("Проклятие распада", cost=30, dmg=30, cd=4, effect=debuff_def("Проклятие", 8, 3)),
    mag("Темпоральный удар", cost=45, dmg=65, cd=4, effect=stun("Темпоральный захват")),
    mag("Уничтожение",       cost=60, dmg=95, cd=5, variance=0.25, crit=0.15),
    mag("Апокалипсис",       cost=80, dmg=120, cd=6, variance=0.30, crit=0.20),
]

# ---------------------------------------------------------------------------
# Сборка полного пула навыков (50 шт.)
# ---------------------------------------------------------------------------
ALL_SKILLS: list[Skill] = (
    WARRIOR_SKILLS
    + PYRO_SKILLS
    + CRYO_SKILLS
    + SHADOW_SKILLS
    + LIGHT_SKILLS
    + CHAOS_SKILLS
)


# ---------------------------------------------------------------------------
# Актёры для масштабного боя
# ---------------------------------------------------------------------------

def massive_player() -> Actor:
    """Игрок с расширенным пулом характеристик для Big Dataset Benchmark."""
    p = Actor(
        name="Игрок",
        hp=300, max_hp=300,
        mp=150, max_mp=150,
        attack=12, defense=6,
        skills=ALL_SKILLS,
    )
    p.init_cooldowns()
    return p


def massive_enemy() -> Actor:
    """Враг с расширенным пулом характеристик для Big Dataset Benchmark."""
    e = Actor(
        name="Враг",
        hp=300, max_hp=300,
        mp=150, max_mp=150,
        attack=12, defense=6,
        skills=ALL_SKILLS,
    )
    e.init_cooldowns()
    return e


# ---------------------------------------------------------------------------
# Главная функция генерации
# ---------------------------------------------------------------------------

def generate() -> RulesPreset:
    """
    Генерирует и валидирует пресет massive_rules.json.

    Returns:
        Валидированный объект RulesPreset.
    """
    config = CombatConfig(
        max_turns=30,
        mp_regen=10,
        is_stochastic=True,
    )

    player = massive_player()
    enemy = massive_enemy()

    preset = RulesPreset(
        name="massive_benchmark",
        combat_config=config,
        player=player,
        enemy=enemy,
        skills=ALL_SKILLS,
    )

    # Pydantic v2 автоматически валидирует при создании.
    # Дополнительная проверка через model_validate для надёжности.
    RulesPreset.model_validate(preset.model_dump())

    save_rules(preset, OUTPUT_PATH)
    return preset


if __name__ == "__main__":
    print("[INFO] Генерация massive_rules.json...")
    p = generate()
    total = len(p.skills)
    print(f"[OK]   Пресет '{p.name}' сохранён в {OUTPUT_PATH}")
    print(f"[OK]   Навыков в пуле: {total}")
    print(f"[OK]   Action Space: Discrete({total + 1})")
    print(f"[OK]   Obs Space: Box(4 + {total} + 6 = {4 + total + 6})")
    print(f"[OK]   HP: {p.player.hp}, MP: {p.player.mp}, "
          f"MP-реген: {p.combat_config.mp_regen}/ход, "
          f"Лимит ходов: {p.combat_config.max_turns}")
    schools = [
        ("Воинское мастерство", len(WARRIOR_SKILLS)),
        ("Пиромантия",          len(PYRO_SKILLS)),
        ("Криомантия",          len(CRYO_SKILLS)),
        ("Тени и Яды",          len(SHADOW_SKILLS)),
        ("Свет и Защита",       len(LIGHT_SKILLS)),
        ("Запретная магия",     len(CHAOS_SKILLS)),
    ]
    print("\n  Школы навыков:")
    for school, count in schools:
        print(f"    {school}: {count} навыков")
