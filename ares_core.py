"""
ares_core.py — Ядро системы ARES: обучение RL-агента, оценка и СППР-анализ.

Содержит три основных модуля:
  1. Обучение MaskablePPO с обёрткой ActionMasker.
  2. Оценка обученного агента против Random и Greedy базовых агентов.
  3. Генерация отчёта СППР с детекцией доминантных стратегий
     и рекомендациями по ребалансу.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from combat_env import CombatEnv
from models import (
    Actor,
    BalanceRecommendation,
    BalanceReport,
    CombatConfig,
    default_config,
    default_enemy,
    default_player,
)


# ---------------------------------------------------------------------------
# Утилита: функция маски для ActionMasker
# ---------------------------------------------------------------------------

def _mask_fn(env: CombatEnv) -> np.ndarray:
    """
    Извлекает маску действий из среды для обёртки ActionMasker.

    Args:
        env: Экземпляр CombatEnv.

    Returns:
        Булев массив допустимых действий.
    """
    return env.action_masks()


# ---------------------------------------------------------------------------
# 1. Обучение агента
# ---------------------------------------------------------------------------

def train_agent(
    player: Actor | None = None,
    enemy: Actor | None = None,
    config: CombatConfig | None = None,
    total_timesteps: int = 20_000,
    model_path: str = "models/ares_agent",
    progress_callback: Callable[[int], None] | None = None,
) -> MaskablePPO:
    """
    Обучает агента MaskablePPO на боевой среде с Action Masking.

    Создаёт среду CombatEnv, оборачивает её в ActionMasker,
    и запускает обучение на заданное число шагов.

    Args:
        player: Шаблон игрока (если None — по умолчанию).
        enemy: Шаблон врага (если None — по умолчанию).
        config: Конфигурация боя.
        total_timesteps: Общее количество шагов обучения.
        model_path: Путь для сохранения обученной модели.
        progress_callback: Опциональный коллбэк для прогресс-бара,
                           вызывается после каждого шага обучения.

    Returns:
        Обученная модель MaskablePPO.
    """
    # Создаём среду и оборачиваем в ActionMasker
    env = CombatEnv(
        player=player or default_player(),
        enemy=enemy or default_enemy(),
        config=config or default_config(),
    )
    env = ActionMasker(env, _mask_fn)

    # Инициализируем MaskablePPO
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Создаём директорию для модели
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    # Обучение
    model.learn(total_timesteps=total_timesteps)

    # Сохраняем модель
    model.save(model_path)

    env.close()
    return model


# ---------------------------------------------------------------------------
# 2. Оценка агента
# ---------------------------------------------------------------------------

def evaluate_agent(
    model: MaskablePPO,
    player: Actor | None = None,
    enemy: Actor | None = None,
    config: CombatConfig | None = None,
    n_episodes: int = 200,
) -> dict[str, Any]:
    """
    Оценивает обученного агента против Random, Greedy и Mirror (Self-Play) агентов.

    Для каждого оппонента прогоняет n_episodes эпизодов и собирает:
      - Win Rate (доля побед агента).
      - Avg TTK (среднее число ходов до завершения боя).
      - Skill Usage (частота применения каждого навыка).

    Args:
        model: Обученная модель MaskablePPO.
        player: Шаблон игрока.
        enemy: Шаблон врага.
        config: Конфигурация боя.
        n_episodes: Количество эпизодов для оценки.

    Returns:
        Словарь с результатами оценки (ключи: random, greedy, mirror).
    """
    results: dict[str, Any] = {}

    for opponent_name in ["random", "greedy", "mirror"]:
        wins = 0
        total_turns: list[int] = []
        skill_counts: dict[int, int] = defaultdict(int)
        total_actions = 0
        total_stun_turns_in_wins = 0
        total_turns_in_wins = 0

        # В зеркальном матче передаём саму модель
        enemy_model = model if opponent_name == "mirror" else None

        for _ in range(n_episodes):
            env = CombatEnv(
                player=player or default_player(),
                enemy=enemy or default_enemy(),
                config=config or default_config(),
                enemy_type=opponent_name,
                enemy_model=enemy_model,
            )
            wrapped_env = ActionMasker(env, _mask_fn)

            obs, _ = wrapped_env.reset()
            done = False
            turns = 0
            enemy_stunned_turns = 0

            while not done:
                # Агент выбирает действие с учётом маски
                action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())

                obs, reward, terminated, truncated, info = wrapped_env.step(int(action))
                done = terminated or truncated
                turns += 1
                
                if info.get("enemy_stunned_this_turn", False):
                    enemy_stunned_turns += 1

                # Подсчёт использования навыков
                if int(action) < env.num_skills:
                    skill_counts[int(action)] += 1
                    total_actions += 1

            # Определяем результат
            if reward > 0:
                wins += 1
                total_stun_turns_in_wins += enemy_stunned_turns
                total_turns_in_wins += turns
            total_turns.append(turns)

            wrapped_env.close()

        # Считаем статистику использования навыков
        skill_usage: dict[int, float] = {}
        if total_actions > 0:
            for action_idx, count in skill_counts.items():
                skill_usage[action_idx] = count / total_actions

        stun_rate_in_wins = total_stun_turns_in_wins / total_turns_in_wins if total_turns_in_wins > 0 else 0.0

        results[opponent_name] = {
            "win_rate": wins / n_episodes * 100,
            "avg_ttk": np.mean(total_turns) if total_turns else 0.0,
            "skill_usage": skill_usage,
            "stun_rate_in_wins": stun_rate_in_wins,
        }

    return results


# ---------------------------------------------------------------------------
# 4. Генерация отчёта СППР
# ---------------------------------------------------------------------------

def generate_balance_report(
    eval_results: dict[str, Any],
    skills_list: list | None = None,
    dominance_threshold: float = 0.40,
    underuse_threshold: float = 0.05,
) -> BalanceReport:
    """
    Генерирует отчёт СППР на основе результатов оценки.

    Алгоритм детекции дисбаланса (трёхуровневый):
      1. Макро-баланс [CRIT_IMBALANCE]:
         Если Avg Win Rate >= 85%, мета признаётся критически
         разбалансированной. Приоритет — нерфы, а не баффы.
      2. Детекция доминирующих навыков:
         a) [DOMINANT]: Pick Rate >= 40% при Avg Win Rate >= 80%.
         b) [EXPLOIT_ROTATION]: два навыка суммарно >= 70% использования
            при Avg TTK <= 5 ходов.
      3. Недоиспользуемые навыки [LOW] — кандидаты на бафф;
         баффы замораживаются, пока активны доминанты.

    Рекомендации по ребалансу:
      - [EXPLOIT_ROTATION]: damage -15% (мин. -2), cooldown +1, cost +3 MP.
      - [DOMINANT]:         damage -15% (мин. -2), cooldown +1.
      - [BUFF]:             damage +10%, cost -20% (заморожен при наличии нерфов).

    Args:
        eval_results: Результаты evaluate_agent().
        skills_list: Список навыков (для получения имён).
        dominance_threshold: Порог Pick Rate для [DOMINANT] (по умолчанию 0.40).
        underuse_threshold: Порог Pick Rate для [LOW] (по умолчанию 0.05).

    Returns:
        Объект BalanceReport с метриками и рекомендациями.
    """
    from models import default_skills
    skills = skills_list or default_skills()

    # Извлекаем метрики из результатов оценки
    random_res = eval_results.get("random", {})
    greedy_res = eval_results.get("greedy", {})
    mirror_res = eval_results.get("mirror", {})

    # Средний Win Rate по базовым оппонентам (random, greedy) для макро-оценки
    avg_win_rate = (
        random_res.get("win_rate", 0.0) + greedy_res.get("win_rate", 0.0)
    ) / 2

    # Средний TTK по базовым оппонентам
    avg_ttk = (
        float(random_res.get("avg_ttk", 0.0)) + float(greedy_res.get("avg_ttk", 0.0))
    ) / 2

    # Объединяем статистику использования навыков по ВСЕМ оппонентам
    combined_usage: dict[str, float] = {}
    raw_random = random_res.get("skill_usage", {})
    raw_greedy = greedy_res.get("skill_usage", {})
    raw_mirror = mirror_res.get("skill_usage", {})

    # Средний Stun Rate в победах
    avg_stun_rate = (
        random_res.get("stun_rate_in_wins", 0.0) + 
        greedy_res.get("stun_rate_in_wins", 0.0) + 
        mirror_res.get("stun_rate_in_wins", 0.0)
    ) / 3

    num_skills = len(skills)
    for i in range(num_skills):
        usage_r = raw_random.get(i, 0.0)
        usage_g = raw_greedy.get(i, 0.0)
        usage_m = raw_mirror.get(i, 0.0)
        # Среднее использование по трём типам оппонентов
        avg_usage = (usage_r + usage_g + usage_m) / 3
        combined_usage[skills[i].name] = round(avg_usage, 4)

    # Добавляем пропуск хода в статистику
    skip_r = raw_random.get(num_skills, 0.0)
    skip_g = raw_greedy.get(num_skills, 0.0)
    skip_m = raw_mirror.get(num_skills, 0.0)
    combined_usage["Пропуск хода"] = round((skip_r + skip_g + skip_m) / 3, 4)

    # ------------------------------------------------------------------
    # Шаг 1: Проверка макро-баланса [CRIT_IMBALANCE]
    # ------------------------------------------------------------------
    crit_imbalance: bool = avg_win_rate >= 85.0

    # ------------------------------------------------------------------
    # Шаг 2: Сортировка навыков по убыванию Pick Rate
    # ------------------------------------------------------------------
    skill_usages: list[tuple[str, float]] = [
        (skills[i].name, combined_usage.get(skills[i].name, 0.0))
        for i in range(num_skills)
    ]
    skill_usages.sort(key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Шаг 3: Детекция эксплойт-ротации [EXPLOIT_ROTATION]
    # ------------------------------------------------------------------
    exploit_rotation_names: set[str] = set()
    if len(skill_usages) >= 3:
        top_3_sum = sum(usage for name, usage in skill_usages[:3])
        random_wr = random_res.get("win_rate", 0.0)
        if top_3_sum >= 0.60 and avg_ttk <= 4.5 and (random_wr >= 80.0 or avg_win_rate >= 60.0):
            for name, usage in skill_usages[:3]:
                if usage >= 0.15:
                    exploit_rotation_names.add(name)
    elif len(skill_usages) >= 2:
        top_2_sum = skill_usages[0][1] + skill_usages[1][1]
        if top_2_sum >= 0.70 and avg_ttk <= 5.0:
            for name, usage in skill_usages[:2]:
                if usage > 0.0:
                    exploit_rotation_names.add(name)

    # ------------------------------------------------------------------
    # Шаг 3.5: Детекция перма-стана [PERMA_STUN_EXPLOIT]
    # ------------------------------------------------------------------
    perma_stun_skills: set[str] = set()
    if avg_stun_rate > 0.40 and avg_win_rate >= 80.0:
        for skill in skills:
            if skill.applied_effect and skill.applied_effect.effect_type == "Оглушение":
                perma_stun_skills.add(skill.name)

    # ------------------------------------------------------------------
    # Шаг 4: Детекция доминирующих навыков [DOMINANT]
    # ------------------------------------------------------------------
    dominant: list[str] = []
    underused: list[str] = []

    for skill in skills:
        usage = combined_usage.get(skill.name, 0.0)

        if skill.name in exploit_rotation_names or skill.name in perma_stun_skills:
            if skill.name not in dominant:
                dominant.append(skill.name)
        elif avg_win_rate >= 80.0 and usage >= dominance_threshold:
            if skill.name not in dominant:
                dominant.append(skill.name)
        else:
            if usage < underuse_threshold:
                underused.append(skill.name)

    # ------------------------------------------------------------------
    # Шаг 5: Генерация рекомендаций
    # ------------------------------------------------------------------
    nerf_recs: list[BalanceRecommendation] = []
    buff_recs: list[BalanceRecommendation] = []

    for skill_name in dominant:
        skill = next(s for s in skills if s.name == skill_name)
        usage = combined_usage.get(skill_name, 0.0)
        is_exploit = skill_name in exploit_rotation_names
        is_perma_stun = skill_name in perma_stun_skills

        damage_delta = -max(2, int(skill.damage * 0.15))

        if is_perma_stun:
            tag = "[PERMA_STUN_EXPLOIT]"
            reason = f"Агент удерживает врага в стане {avg_stun_rate*100:.1f}% времени"
            cooldown_delta = 1
            cost_delta = 0
            effect_duration_delta = -1
        elif is_exploit:
            tag = "[EXPLOIT_ROTATION]"
            reason = (
                f"{tag} Навык «{skill_name}» входит в доминирующую ротацию. "
                f"Pick Rate: {usage:.1%}, Avg WR: {avg_win_rate:.1f}%, "
                f"Avg TTK: {avg_ttk:.1f} ходов."
            )
            cooldown_delta = 1
            cost_delta = 3
            effect_duration_delta = 0
        else:
            tag = "[DOMINANT]"
            reason = (
                f"{tag} Навык «{skill_name}» доминирует в мете. "
                f"Pick Rate: {usage:.1%}, Avg WR: {avg_win_rate:.1f}%."
            )
            cooldown_delta = 1
            cost_delta = 0
            effect_duration_delta = 0

        # Добавляем плашку критического дисбаланса, если мета разрушена
        if crit_imbalance:
            reason = f"[CRIT_IMBALANCE] {reason}"

        nerf_recs.append(
            BalanceRecommendation(
                skill_name=skill_name,
                reason=reason,
                damage_delta=damage_delta,
                cooldown_delta=cooldown_delta,
                cost_delta=cost_delta,
                effect_duration_delta=effect_duration_delta,
            )
        )



    # --- Рекомендации [BUFF] для недоиспользуемых навыков ---
    # ВАЖНО: баффы допустимы только после устранения доминирующей ротации.
    top_3_sum = sum(usage for name, usage in skill_usages[:3]) if len(skill_usages) >= 3 else 0.0

    if avg_win_rate >= 75.0:
        buff_recs.append(
            BalanceRecommendation(
                skill_name="Система",
                reason="[BLOCKED: High Macro WinRate] Общий винрейт >= 75%. Генерация баффов заблокирована до устранения текущих доминантных стратегий.",
                damage_delta=0,
                cooldown_delta=0,
                cost_delta=0,
            )
        )
    elif top_3_sum >= 0.60:
        buff_recs.append(
            BalanceRecommendation(
                skill_name="Система",
                reason="[BLOCKED: Dominant Rotation Active] Баффы заморожены: обнаружена монополия ротации (Топ-3 занимают >60% действий).",
                damage_delta=0,
                cooldown_delta=0,
                cost_delta=0,
            )
        )
    else:
        for skill_name in underused:
            skill = next(s for s in skills if s.name == skill_name)
            usage = combined_usage.get(skill_name, 0.0)
            cost_delta = -max(1, int(skill.cost * 0.20)) if skill.cost > 0 else 0
            damage_delta = max(1, int(skill.damage * 0.10))

            # Бафф замораживается пока активны доминанты
            if dominant:
                prefix = (
                    "[BUFF | ЗАМОРОЖЕНО: применять только после нерфа доминирующей ротации]"
                )
            else:
                prefix = "[BUFF]"

            reason = (
                f"{prefix} Навык «{skill_name}» не используется "
                f"(Pick Rate: {usage:.1%}). "
                f"Рекомендован бафф: урон +{damage_delta} ед."
            )
            if cost_delta < 0:
                reason += f", стоимость {cost_delta} MP."

            buff_recs.append(
                BalanceRecommendation(
                    skill_name=skill_name,
                    reason=reason,
                    damage_delta=damage_delta,
                    cooldown_delta=0,
                    cost_delta=cost_delta,
                )
            )

    # Нерфы первыми, баффы — вторыми
    recommendations = nerf_recs + buff_recs

    return BalanceReport(
        win_rate_vs_random=round(random_res.get("win_rate", 0.0), 2),
        win_rate_vs_greedy=round(greedy_res.get("win_rate", 0.0), 2),
        win_rate_vs_mirror=round(mirror_res.get("win_rate", 0.0), 2),
        avg_ttk_vs_random=round(float(random_res.get("avg_ttk", 0.0)), 2),
        avg_ttk_vs_greedy=round(float(greedy_res.get("avg_ttk", 0.0)), 2),
        avg_ttk_vs_mirror=round(float(mirror_res.get("avg_ttk", 0.0)), 2),
        skill_usage=combined_usage,
        dominant_skills=dominant,
        underused_skills=underused,
        recommendations=recommendations,
    )

