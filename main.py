"""
main.py — Интерактивный терминальный интерфейс ARES на библиотеке Rich.

Точка входа приложения. Предоставляет:
  1. Запуск стресс-тестирования баланса (обучение + оценка + отчёт).
  2. Просмотр аналитического отчёта СППР.
  3. Сценарный анализ «Что, если?»: ручная правка параметров навыка.
  4. Экспорт отчёта в файл (Markdown / JSON).
  5. Загрузка конфигурации правил из внешнего JSON-файла.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from models import (
    BalanceReport,
    Skill,
    SkillType,
    default_config,
    default_enemy,
    default_player,
    default_skills,
)
from rules_manager import (
    DEFAULT_RULES_PATH,
    RulesPreset,
    apply_balance_recommendations,
    ensure_default_rules,
    load_rules,
    save_rules,
)

# Консоль Rich
console = Console()

# Глобальные переменные состояния
_current_report: BalanceReport | None = None
_current_preset: RulesPreset | None = None  # Активный пресет правил сессии


# =========================================================================
# ASCII-баннер
# =========================================================================

BANNER = r"""
[bold cyan]
     █████╗ ██████╗ ███████╗███████╗
    ██╔══██╗██╔══██╗██╔════╝██╔════╝
    ███████║██████╔╝█████╗  ███████╗
    ██╔══██║██╔══██╗██╔══╝  ╚════██║
    ██║  ██║██║  ██║███████╗███████║
    ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
[/bold cyan]
[bold white]  Adversarial Roleplaying Equilibrium System[/bold white]
[dim]  Система Поддержки Баланса RPG • v0.1[/dim]
"""


def show_banner() -> None:
    """Отображает стилизованный ASCII-баннер ARES."""
    console.print(
        Panel(
            Align.center(BANNER),
            border_style="bright_cyan",
            box=box.DOUBLE_EDGE,
            padding=(0, 2),
        )
    )


# =========================================================================
# Главное меню
# =========================================================================

def show_menu() -> str:
    """
    Отображает главное меню и возвращает выбор пользователя.

    Returns:
        Строка с номером выбранного пункта.
    """
    menu_text = (
        "[bold green]1[/] | Запустить стресс-тестирование баланса\n"
        "[bold green]2[/] | Посмотреть отчёт СППР\n"
        "[bold green]3[/] | Сценарный анализ «Что, если?»\n"
        "[bold green]4[/] | Экспорт отчёта в файл\n"
        "[bold green]5[/] | Загрузить конфигурацию правил из файла (JSON)\n"
        "[bold green]6[/] | Применить рекомендации СППР (Авто-ребаланс)\n"
        "[bold red]0[/]   | Выход"
    )
    console.print(
        Panel(
            menu_text,
            title="[bold]═══ ГЛАВНОЕ МЕНЮ ═══[/bold]",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    choice = Prompt.ask(
        "[bold yellow]Выберите действие[/bold yellow]",
        choices=["0", "1", "2", "3", "4", "5", "6"],
        default="0",
    )
    return choice


# =========================================================================
# 1. Стресс-тестирование баланса
# =========================================================================

def run_stress_test(
    preset: RulesPreset | None = None,
    skills: list[Skill] | None = None,
) -> BalanceReport:
    """
    Запускает полный цикл стресс-тестирования:
      1. Обучение MaskablePPO с прогресс-баром.
      2. Оценка против Random, Greedy и Mirror агентов.
      3. Генерация отчёта СППР.

    Args:
        preset: Активный пресет правил (если None — глобальный _current_preset).
        skills: Переопределённый список навыков (для режима «Что, если?»).

    Returns:
        Объект BalanceReport с результатами.
    """
    global _current_preset
    # Ленивый импорт, чтобы не замедлять загрузку меню
    from ares_core import evaluate_agent, generate_balance_report, train_agent

    # Берём активный пресет
    active_preset = preset or _current_preset or ensure_default_rules()

    if skills is not None:
        # В режиме «Что, если?» навыки переопределяются, остальное из пресета
        player = active_preset.player.model_copy(deep=True)
        player.skills = [s.model_copy(deep=True) for s in skills]
        player.init_cooldowns()

        enemy = active_preset.enemy.model_copy(deep=True)
        enemy.skills = [s.model_copy(deep=True) for s in skills]
        enemy.init_cooldowns()

        current_skills = skills
    else:
        player = active_preset.build_player()
        enemy = active_preset.build_enemy()
        current_skills = active_preset.skills

    config = active_preset.combat_config
    total_steps = 20_000

    console.print()

    # --- Фаза 1: Обучение ---
    console.print(
        Panel(
            "[bold]Фаза 1/3: Обучение MaskablePPO[/bold]\n"
            f"Шаги обучения: {total_steps:,}",
            title="[ФАЗА 1] Обучение агента",
            border_style="blue",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Обучение MaskablePPO...", total=100)

        # Запускаем обучение (прогресс-бар имитирует ход обучения)
        start_time = time.time()
        model = train_agent(
            player=player,
            enemy=enemy,
            config=config,
            total_timesteps=total_steps,
        )
        elapsed = time.time() - start_time

        # Завершаем прогресс-бар
        progress.update(task, completed=100)

    console.print(
        f"  [green][OK][/green] Обучение завершено за {elapsed:.1f} сек.\n"
    )

    # --- Фаза 2: Оценка ---
    console.print(
        Panel(
            "[bold]Фаза 2/3: Оценка агента[/bold]\n"
            "Прогон 200 эпизодов vs Random и Greedy",
            title="[ФАЗА 2] Оценка",
            border_style="yellow",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow]{task.description}"),
        BarColumn(bar_width=40),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Оценка агента...", total=None)

        eval_results = evaluate_agent(
            model=model,
            player=player,
            enemy=enemy,
            config=config,
            n_episodes=200,
        )

        progress.update(task, completed=100, total=100)

    console.print("  [green][OK][/green] Оценка завершена.\n")

    # --- Фаза 3: Анализ СППР ---
    console.print(
        Panel(
            "[bold]Фаза 3/3: Анализ данных СППР[/bold]\n"
            "Детекция дисбаланса и формирование рекомендаций",
            title="[ФАЗА 3] Анализ",
            border_style="magenta",
        )
    )

    report = generate_balance_report(
        eval_results=eval_results,
        skills_list=current_skills,
    )

    console.print("  [green][OK][/green] Отчёт СППР сформирован.\n")

    return report


# =========================================================================
# 2. Отображение отчёта СППР
# =========================================================================

def show_report(report: BalanceReport) -> None:
    """
    Отображает отчёт СППР в виде красочных таблиц Rich.

    Выводит:
      - Таблицу Win Rate и TTK по оппонентам.
      - Таблицу использования навыков.
      - Список доминантных и недоиспользуемых навыков.
      - Рекомендации по ребалансу.

    Args:
        report: Объект BalanceReport.
    """
    console.print()

    # --- Таблица метрик ---
    metrics_table = Table(
        title="Метрики баланса",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    metrics_table.add_column("Оппонент", style="bold")
    metrics_table.add_column("Win Rate", justify="center")
    metrics_table.add_column("Средний TTK (ходов)", justify="center")

    # Определяем цвет Win Rate
    def wr_color(wr: float) -> str:
        if wr >= 70:
            return "green"
        elif wr >= 40:
            return "yellow"
        return "red"

    metrics_table.add_row(
        "vs Random",
        f"[{wr_color(report.win_rate_vs_random)}]{report.win_rate_vs_random:.1f}%[/]",
        f"{report.avg_ttk_vs_random:.1f}",
    )
    metrics_table.add_row(
        "vs Greedy",
        f"[{wr_color(report.win_rate_vs_greedy)}]{report.win_rate_vs_greedy:.1f}%[/]",
        f"{report.avg_ttk_vs_greedy:.1f}",
    )
    metrics_table.add_row(
        "vs Self-Play (Mirror)",
        f"[{wr_color(report.win_rate_vs_mirror)}]{report.win_rate_vs_mirror:.1f}%[/]",
        f"{report.avg_ttk_vs_mirror:.1f}",
    )
    console.print(metrics_table)
    console.print()

    # --- Таблица использования навыков ---
    usage_table = Table(
        title="Частота использования навыков",
        box=box.ROUNDED,
        border_style="yellow",
        show_header=True,
        header_style="bold yellow",
    )
    usage_table.add_column("Навык", style="bold")
    usage_table.add_column("Доля использования", justify="center")
    usage_table.add_column("Статус", justify="center")

    for skill_name, usage in report.skill_usage.items():
        # Определяем статус
        if skill_name in report.dominant_skills:
            status = "[bold red][DOMINANT][/]"
        elif skill_name in report.underused_skills:
            status = "[dim][LOW][/]"
        else:
            status = "[green][OK][/]"

        # Полоска-визуализация
        bar_len = int(usage * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)

        usage_table.add_row(
            skill_name,
            f"{bar} {usage:.1%}",
            status,
        )

    console.print(usage_table)
    console.print()

    # --- Рекомендации ---
    if report.recommendations:
        rec_table = Table(
            title="Рекомендации по ребалансу",
            box=box.ROUNDED,
            border_style="magenta",
            show_header=True,
            header_style="bold magenta",
        )
        rec_table.add_column("Навык", style="bold")
        rec_table.add_column("Дельта урона", justify="center")
        rec_table.add_column("Дельта КД", justify="center")
        rec_table.add_column("Дельта стоимости", justify="center")
        rec_table.add_column("Обоснование", max_width=50)

        for rec in report.recommendations:
            # Форматируем дельты с цветом
            def fmt_delta(val: int) -> str:
                if val > 0:
                    return f"[green]+{val}[/]"
                elif val < 0:
                    return f"[red]{val}[/]"
                return "[dim]0[/]"

            rec_table.add_row(
                rec.skill_name,
                fmt_delta(rec.damage_delta),
                fmt_delta(rec.cooldown_delta),
                fmt_delta(rec.cost_delta),
                rec.reason,
            )

        console.print(rec_table)
    else:
        console.print(
            Panel(
                "[green]Баланс в пределах нормы. "
                "Доминантных или недоиспользуемых навыков не обнаружено.[/green]",
                title="[OK] Заключение",
                border_style="green",
            )
        )

    console.print()

    # --- Графовый анализ синергий ---
    if getattr(report, "synergy_cycles", None) or getattr(report, "key_synergy_nodes", None):
        console.print(
            Panel(
                "[bold cyan]Анализ синергий и графовые метрики[/bold cyan]",
                border_style="cyan"
            )
        )
        
        if report.synergy_cycles:
            console.print("[bold yellow]Обнаружены циклические ротации (абьюз-лупы):[/bold yellow]")
            for cycle, min_weight in report.synergy_cycles:
                chain_str = " -> ".join(f"[{node}]" for node in cycle)
                console.print(f"  {chain_str} [dim](вес: {min_weight})[/dim]")
            console.print()

        if report.key_synergy_nodes:
            console.print("[bold yellow]Топ узлов по Betweenness Centrality (мосты):[/bold yellow]")
            for i, (node, score) in enumerate(report.key_synergy_nodes):
                console.print(f"  {i+1}. [bold cyan]{node}[/bold cyan] [red][KEY_SYNERGY_NODE][/red] (score: {score:.3f})")
        console.print()


# =========================================================================
# 3. Сценарный анализ «Что, если?»
# =========================================================================

def _parse_value(raw: str, current: int) -> int:
    """
    Парсер ввода значений параметров навыков.

    Логика:
      - Строка начинается с '+' или '-': значение трактуется как дельта.
        new = max(0, current + delta)
      - Обычное число без знака: абсолютная перезапись.
        new = max(0, val)
      - Пустая строка: сохраняется текущее значение.

    Args:
        raw: Сырая строка ввода пользователя.
        current: Текущее значение параметра.

    Returns:
        Новое целочисленное значение >= 0.
    """
    raw = raw.strip()
    if not raw:
        return current
    if raw.startswith(("+", "-")):
        try:
            delta = int(raw)
            return max(0, current + delta)
        except ValueError:
            return current
    try:
        return max(0, int(raw))
    except ValueError:
        return current


def what_if_analysis() -> BalanceReport | None:
    """
    Интерактивный режим сценарного анализа «Что, если?».

    Использует навыки из активного пресета (_current_preset).
    Поддерживает ввод дельт (+3, -5) и абсолютных значений (22).
    После редактирования предлагает сохранить изменения в JSON-файл.

    Returns:
        Новый отчёт BalanceReport или None при отмене.
    """
    global _current_preset

    # Берём навыки из активного пресета (не из дефолтов!)
    active_preset = _current_preset or ensure_default_rules()
    skills = [s.model_copy(deep=True) for s in active_preset.skills]

    console.print(
        Panel(
            "[bold]Режим сценарного анализа «Что, если?»[/bold]\n"
            "Активный пресет: [cyan]{name}[/cyan]\n"
            "Введите дельту (+3, -5) или абсолютное значение (22). Пусто — без изменений.".format(
                name=active_preset.name
            ),
            title="Сценарный анализ",
            border_style="cyan",
        )
    )
    console.print()

    # Таблица текущих навыков
    skill_table = Table(
        title="Текущие параметры навыков",
        box=box.SIMPLE_HEAVY,
        border_style="cyan",
    )
    skill_table.add_column("#", style="bold", justify="center")
    skill_table.add_column("Навык", style="bold")
    skill_table.add_column("Урон", justify="center")
    skill_table.add_column("Стоимость", justify="center")
    skill_table.add_column("Кулдаун", justify="center")
    skill_table.add_column("Тип", justify="center")
    skill_table.add_column("Эффект", justify="left")

    for i, s in enumerate(skills):
        effect_str = "—"
        if s.applied_effect:
            eff_type = s.applied_effect.effect_type
            if eff_type == "Оглушение":
                effect_str = f"[STUN] {s.applied_effect.duration}х"
            elif eff_type == "ДоТ (периодический урон)":
                effect_str = f"[DOT] {s.applied_effect.value} ед. {s.applied_effect.duration}х"
            elif eff_type == "Щит":
                effect_str = f"[SHIELD] {s.applied_effect.value} ед. {s.applied_effect.duration}х"
            elif eff_type == "Усиление атаки":
                effect_str = f"[BUFF_ATK] +{s.applied_effect.value} {s.applied_effect.duration}х"
            elif eff_type == "Срез брони":
                effect_str = f"[DEBUFF_DEF] -{s.applied_effect.value} {s.applied_effect.duration}х"

        skill_table.add_row(
            str(i + 1),
            s.name,
            str(s.damage),
            str(s.cost),
            str(s.cooldown),
            s.skill_type.value,
            effect_str,
        )

    console.print(skill_table)
    console.print()

    # Выбор навыка
    choices = [str(i + 1) for i in range(len(skills))] + ["0"]
    idx_str = Prompt.ask(
        "[bold yellow]Выберите номер навыка для редактирования (0 — отмена)[/]",
        choices=choices,
        default="0",
    )

    if idx_str == "0":
        console.print("[dim]Отмена сценарного анализа.[/dim]")
        return None

    skill_idx = int(idx_str) - 1
    skill = skills[skill_idx]

    console.print(
        f"\n[bold]Редактирование навыка: [cyan]{skill.name}[/cyan][/bold]\n"
        f"Введите дельту ([bold]+3[/bold], [bold]-5[/bold]) или "
        f"абсолютное значение ([bold]22[/bold]). Пусто — без изменений.\n"
    )

    # Ввод новых параметров с поддержкой дельт
    raw_damage = Prompt.ask(
        f"  Урон [dim](текущий: {skill.damage})[/dim]",
        default="",
    )
    raw_cost = Prompt.ask(
        f"  Стоимость маны [dim](текущая: {skill.cost})[/dim]",
        default="",
    )
    raw_cd = Prompt.ask(
        f"  Кулдаун [dim](текущий: {skill.cooldown})[/dim]",
        default="",
    )

    new_damage = _parse_value(raw_damage, skill.damage)
    new_cost = _parse_value(raw_cost, skill.cost)
    new_cd = _parse_value(raw_cd, skill.cooldown)

    # Создаём модифицированный навык
    skills[skill_idx] = Skill(
        name=skill.name,
        damage=new_damage,
        cost=new_cost,
        cooldown=new_cd,
        skill_type=skill.skill_type,
        crit_chance=skill.crit_chance,
        damage_variance=skill.damage_variance,
        applied_effect=skill.applied_effect,
        target_self=skill.target_self,
    )

    console.print(
        Panel(
            f"[bold green]Навык «{skill.name}» изменён:[/bold green]\n"
            f"  Урон:       {skill.damage} → {skills[skill_idx].damage}\n"
            f"  Стоимость:  {skill.cost} → {skills[skill_idx].cost}\n"
            f"  Кулдаун:    {skill.cooldown} → {skills[skill_idx].cooldown}",
            border_style="green",
        )
    )

    # Предложение сохранить изменения в активный JSON-файл
    save_choice = Prompt.ask(
        "\n[bold yellow]Сохранить изменения в активный JSON-конфиг?[/bold yellow] [dim][y/N][/dim]",
        default="n",
    )
    if save_choice.lower() == "y":
        # Обновляем пресет и сохраняем
        updated_preset = active_preset.model_copy(deep=True)
        updated_preset.skills = skills
        save_rules(updated_preset, DEFAULT_RULES_PATH)
        _current_preset = updated_preset
        console.print(
            f"  [green][OK][/green] Изменения сохранены в [cyan]{DEFAULT_RULES_PATH.name}[/cyan]"
        )
    else:
        console.print("  [dim][INFO] Изменения применяются только для текущего прогона.[/dim]")

    console.print("\n[bold]Запуск повторного стресс-тестирования...[/bold]\n")

    report = run_stress_test(skills=skills)
    return report


# =========================================================================
# 4. Экспорт отчёта
# =========================================================================

def export_report(report: BalanceReport) -> None:
    """
    Экспортирует отчёт СППР в файл (Markdown или JSON).

    Args:
        report: Объект BalanceReport для экспорта.
    """
    console.print(
        Panel(
            "[bold green]1[/] │ Markdown (.md)\n"
            "[bold green]2[/] │ JSON (.json)",
            title="Формат экспорта",
            border_style="green",
        )
    )

    fmt = Prompt.ask(
        "[bold yellow]Выберите формат[/]",
        choices=["1", "2"],
        default="1",
    )

    if fmt == "1":
        _export_markdown(report)
    else:
        _export_json(report)


def _export_markdown(report: BalanceReport) -> None:
    """Экспортирует отчёт в формате Markdown."""
    lines: list[str] = [
        "# Отчёт СППР — ARES",
        "",
        "## Метрики баланса",
        "",
        "| Оппонент | Win Rate | Средний TTK |",
        "|----------|----------|-------------|",
        f"| vs Random | {report.win_rate_vs_random:.1f}% | {report.avg_ttk_vs_random:.1f} ходов |",
        f"| vs Greedy | {report.win_rate_vs_greedy:.1f}% | {report.avg_ttk_vs_greedy:.1f} ходов |",
        f"| vs Self-Play (Mirror) | {report.win_rate_vs_mirror:.1f}% | {report.avg_ttk_vs_mirror:.1f} ходов |",
        "",
        "## Использование навыков",
        "",
        "| Навык | Доля |",
        "|-------|------|",
    ]

    for name, usage in report.skill_usage.items():
        lines.append(f"| {name} | {usage:.1%} |")

    lines.append("")

    if report.dominant_skills:
        lines.append("## [CRIT] Доминантные навыки")
        lines.append("")
        for s in report.dominant_skills:
            lines.append(f"- **{s}**")
        lines.append("")

    if report.underused_skills:
        lines.append("## [LOW] Недоиспользуемые навыки")
        lines.append("")
        for s in report.underused_skills:
            lines.append(f"- {s}")
        lines.append("")

    if report.recommendations:
        lines.append("## Рекомендации по ребалансу")
        lines.append("")
        for rec in report.recommendations:
            lines.append(f"### {rec.skill_name}")
            lines.append(f"- Дельта урона: **{rec.damage_delta:+d}**")
            lines.append(f"- Дельта кулдауна: **{rec.cooldown_delta:+d}**")
            lines.append(f"- Дельта стоимости: **{rec.cost_delta:+d}**")
            lines.append(f"- Обоснование: {rec.reason}")
            lines.append("")

    path = Path("report.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    console.print(
        f"[green][OK] Отчёт сохранён в файл: [bold]{path.resolve()}[/bold][/green]"
    )


def _export_json(report: BalanceReport) -> None:
    """Экспортирует отчёт в формате JSON (используя Pydantic v2 model_dump)."""
    path = Path("report.json")
    data = report.model_dump()
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(
        f"[green][OK] Отчёт сохранён в файл: [bold]{path.resolve()}[/bold][/green]"
    )


# =========================================================================
# 5. Загрузка конфигурации правил из JSON-файла
# =========================================================================

def load_rules_interactive() -> bool:
    """
    Интерактивная загрузка пресета правил из пользовательского JSON-файла.

    Запрашивает путь к файлу, валидирует его через RulesPreset,
    обновляет глобальный _current_preset и сообщает о результате.
    Перезагрузка среды CombatEnv произойдёт автоматически при следующем
    вызове run_stress_test(), который считает обновлённый пресет.

    Returns:
        True — если пресет успешно загружен, False — при ошибке или отмене.
    """
    global _current_preset

    console.print(
        Panel(
            "[bold]Загрузка конфигурации правил из JSON-файла[/bold]\n"
            f"Текущий активный пресет: [cyan]{_current_preset.name if _current_preset else 'default'}[/cyan]\n"
            f"Путь по умолчанию: [dim]{DEFAULT_RULES_PATH}[/dim]",
            title="[5] Загрузка правил",
            border_style="cyan",
        )
    )
    console.print()

    path_str = Prompt.ask(
        "[bold yellow]Путь к JSON-файлу правил[/bold yellow] "
        f"[dim](Enter — загрузить {DEFAULT_RULES_PATH.name})[/dim]",
        default=str(DEFAULT_RULES_PATH),
    )

    if not path_str.strip():
        path_str = str(DEFAULT_RULES_PATH)

    path = Path(path_str.strip())

    if not path.exists():
        console.print(
            Panel(
                f"[red]Файл не найден: {path}[/red]",
                title="[ERROR] Ошибка загрузки",
                border_style="red",
            )
        )
        return False

    try:
        preset = load_rules(path)
        _current_preset = preset

        # Формируем сводку загруженного пресета
        skill_list = "\n".join(
            f"  [dim]•[/dim] {s.name} | урон: {s.damage} | стоимость: {s.cost} | КД: {s.cooldown}"
            for s in preset.skills
        )

        console.print(
            Panel(
                f"[bold green][OK] Пресет «{preset.name}» успешно загружен.[/bold green]\n\n"
                f"[bold]Боевая конфигурация:[/bold]\n"
                f"  Лимит ходов:  {preset.combat_config.max_turns}\n"
                f"  Реген маны:   {preset.combat_config.mp_regen} MP/ход\n"
                f"  Стохастика:   {'вкл.' if preset.combat_config.is_stochastic else 'выкл.'}\n\n"
                f"[bold]Навыки ({len(preset.skills)} шт.):[/bold]\n"
                f"{skill_list}\n\n"
                f"[dim][INFO] CombatEnv будет перезагружена автоматически при следующем\n"
                f"запуске стресс-тестирования (пункт 1).[/dim]",
                title=f"[OK] Пресет загружен: {path.name}",
                border_style="green",
            )
        )
        return True

    except Exception as exc:
        console.print(
            Panel(
                f"[red]Ошибка при разборе файла:[/red]\n{exc}",
                title="[ERROR] Ошибка валидации JSON",
                border_style="red",
            )
        )
        return False


# =========================================================================
# 6. Автоматическое применение рекомендаций СППР (Авто-ребаланс)
# =========================================================================

def auto_rebalance_interactive() -> None:
    """
    Интерактивный интерфейс авто-ребаланса.
    Считывает рекомендации из текущего отчета, предлагает режимы (нерфы/все),
    применяет патч к пресету и запускает ретест.
    """
    global _current_report, _current_preset

    if _current_report is None or not _current_report.recommendations:
        console.print(
            Panel(
                "[yellow]Нет доступных рекомендаций для авто-ребаланса.[/yellow]\n"
                "Сначала запустите стресс-тестирование (пункт 1).",
                title="[INFO] Авто-ребаланс",
                border_style="yellow",
            )
        )
        return

    console.print(
        Panel(
            "[bold]Автоматическое применение рекомендаций СППР[/bold]\n"
            "Выберите режим авто-патча:",
            title="[6] Авто-ребаланс",
            border_style="magenta",
        )
    )

    console.print("  [bold green]1[/] | Безопасный режим (только нерфы доминантных ротаций)")
    console.print("  [bold yellow]2[/] | Полный ребаланс (включая баффы неиспользуемых)")
    console.print("  [bold red]0[/] | Отмена\n")

    mode = Prompt.ask(
        "[bold cyan]Выберите режим[/bold cyan]",
        choices=["0", "1", "2"],
        default="1",
    )

    if mode == "0":
        console.print("[dim]Отмена авто-ребаланса.[/dim]")
        return

    apply_frozen_buffs = (mode == "2")
    
    active_preset = _current_preset or ensure_default_rules()

    new_preset, diff_log = apply_balance_recommendations(
        preset=active_preset,
        report=_current_report,
        apply_frozen_buffs=apply_frozen_buffs
    )

    if not diff_log:
        console.print("[yellow]Нет изменений для применения в выбранном режиме.[/yellow]")
        return

    # Выводим таблицу изменений
    diff_table = Table(
        title="Изменения в авто-патче",
        box=box.SIMPLE_HEAVY,
        border_style="magenta",
    )
    diff_table.add_column("Изменённые навыки", style="bold")
    for diff in diff_log:
        diff_table.add_row(diff)
    
    console.print(diff_table)

    save_choice = Prompt.ask(
        "\n[bold yellow]Сохранить авто-патч в текущий JSON-конфиг и запустить повторное тестирование?[/bold yellow] [dim][Y/n][/dim]",
        default="y",
    )

    if save_choice.lower() == "y":
        save_rules(new_preset, DEFAULT_RULES_PATH)
        _current_preset = new_preset
        console.print(f"  [green][OK][/green] Патч успешно сохранён в [cyan]{DEFAULT_RULES_PATH.name}[/cyan]")
        console.print("\n[bold]Запуск повторного стресс-тестирования...[/bold]\n")
        _current_report = run_stress_test()
        show_report(_current_report)
    else:
        console.print("[dim]Отмена сохранения.[/dim]")


# =========================================================================
# Главный цикл приложения
# =========================================================================

def main() -> None:
    """
    Точка входа ARES.

    Отображает баннер и запускает интерактивный цикл главного меню.
    """
    global _current_report, _current_preset

    show_banner()

    # Загружаем активный пресет правил при старте
    console.print(
        Panel(
            "[dim]Загрузка конфигурации правил...[/dim]",
            border_style="dim",
            box=box.ROUNDED,
        )
    )
    try:
        _current_preset = ensure_default_rules()
        console.print(
            Panel(
                f"[bold green]Система готова к работе.[/bold green]\n"
                f"Активный пресет: [cyan]{_current_preset.name}[/cyan] | "
                f"Навыков: {len(_current_preset.skills)} | "
                f"Файл: [dim]{DEFAULT_RULES_PATH.name}[/dim]",
                border_style="green",
                box=box.ROUNDED,
            )
        )
    except Exception as exc:
        console.print(
            f"[yellow][WARN] Не удалось загрузить {DEFAULT_RULES_PATH.name}: {exc}. "
            f"Используются дефолтные параметры.[/yellow]"
        )

    while True:
        console.print()
        choice = show_menu()

        if choice == "0":
            console.print(
                "\n[bold cyan]До встречи! Да пребудет баланс с вами.[/bold cyan]\n"
            )
            break

        elif choice == "1":
            # Стресс-тестирование с активным пресетом
            try:
                _current_report = run_stress_test()
                show_report(_current_report)
            except Exception as e:
                console.print(f"[red]Ошибка при тестировании: {e}[/red]")

        elif choice == "2":
            # Просмотр отчёта
            if _current_report is None:
                console.print(
                    Panel(
                        "[yellow]Отчёт ещё не сформирован.[/yellow]\n"
                        "Сначала запустите стресс-тестирование (пункт 1).",
                        title="[INFO] Внимание",
                        border_style="yellow",
                    )
                )
            else:
                show_report(_current_report)

        elif choice == "3":
            # Сценарный анализ
            try:
                new_report = what_if_analysis()
                if new_report is not None:
                    _current_report = new_report
                    show_report(_current_report)
            except Exception as e:
                console.print(f"[red]Ошибка при сценарном анализе: {e}[/red]")

        elif choice == "4":
            # Экспорт
            if _current_report is None:
                console.print(
                    Panel(
                        "[yellow]Нечего экспортировать.[/yellow]\n"
                        "Сначала запустите стресс-тестирование (пункт 1).",
                        title="[INFO] Внимание",
                        border_style="yellow",
                    )
                )
            else:
                export_report(_current_report)

        elif choice == "5":
            # Загрузка правил из JSON
            try:
                load_rules_interactive()
            except Exception as e:
                console.print(f"[red]Ошибка при загрузке правил: {e}[/red]")

        elif choice == "6":
            # Авто-ребаланс
            try:
                auto_rebalance_interactive()
            except Exception as e:
                console.print(f"[red]Ошибка при авто-ребалансе: {e}[/red]")

if __name__ == "__main__":
    main()
