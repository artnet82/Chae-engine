"""
examples/basic_usage.py

Демо CHAE на открытых данных интернет-магазина.
Строим причинный граф: реклама → трафик → заказы → выручка,
а также влияние цены, сезона и возвратов.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from chae import CHAE, CHAEConfig


def load_open_shop_data() -> List[Dict[str, Any]]:
    """
    Небольшие открытые (синтетические, но реалистичные) данные
    интернет-магазина за 12 недель.
    Можно заменить на любой публичный CSV/JSON.
    """
    return [
        {"week": 1,  "ads_spend": 120, "price": 29.9, "season": "low",  "traffic": 820,  "orders": 41,  "returns": 4,  "revenue": 1100},
        {"week": 2,  "ads_spend": 150, "price": 29.9, "season": "low",  "traffic": 910,  "orders": 48,  "returns": 5,  "revenue": 1280},
        {"week": 3,  "ads_spend": 180, "price": 27.9, "season": "low",  "traffic": 1100, "orders": 62,  "returns": 6,  "revenue": 1550},
        {"week": 4,  "ads_spend": 90,  "price": 29.9, "season": "low",  "traffic": 700,  "orders": 33,  "returns": 3,  "revenue": 890},
        {"week": 5,  "ads_spend": 200, "price": 24.9, "season": "high", "traffic": 1450, "orders": 95,  "returns": 9,  "revenue": 2100},
        {"week": 6,  "ads_spend": 220, "price": 24.9, "season": "high", "traffic": 1600, "orders": 110, "returns": 11, "revenue": 2450},
        {"week": 7,  "ads_spend": 160, "price": 26.9, "season": "high", "traffic": 1300, "orders": 88,  "returns": 8,  "revenue": 1980},
        {"week": 8,  "ads_spend": 140, "price": 29.9, "season": "high", "traffic": 1050, "orders": 70,  "returns": 7,  "revenue": 1750},
        {"week": 9,  "ads_spend": 100, "price": 31.9, "season": "low",  "traffic": 650,  "orders": 28,  "returns": 2,  "revenue": 780},
        {"week": 10, "ads_spend": 190, "price": 27.9, "season": "low",  "traffic": 1200, "orders": 75,  "returns": 6,  "revenue": 1820},
        {"week": 11, "ads_spend": 210, "price": 25.9, "season": "high", "traffic": 1550, "orders": 105, "returns": 10, "revenue": 2350},
        {"week": 12, "ads_spend": 170, "price": 26.9, "season": "high", "traffic": 1400, "orders": 92,  "returns": 9,  "revenue": 2050},
    ]


def seed_shop_causal_graph(engine: CHAE) -> None:
    """Заранее закладываем здравые причинно-следственные связи магазина."""
    g = engine.causal.graph

    # Реклама → трафик → заказы → выручка
    g.add_cause("ads_spend", "traffic", strength=0.75, evidence="open shop data")
    g.add_cause("traffic", "orders", strength=0.80, evidence="open shop data")
    g.add_cause("orders", "revenue", strength=0.90, evidence="open shop data")

    # Цена влияет на заказы и возвраты
    g.add_cause("price", "orders", strength=-0.55, evidence="higher price → fewer orders")
    g.add_cause("price", "returns", strength=0.30, evidence="higher price → more returns")

    # Сезон влияет на трафик
    g.add_cause("season_high", "traffic", strength=0.65, evidence="high season boost")

    # Возвраты немного уменьшают чистую выручку
    g.add_cause("returns", "revenue", strength=-0.25, evidence="returns reduce net revenue")

    # Заказы тоже немного влияют на возвраты
    g.add_cause("orders", "returns", strength=0.40, evidence="more orders → more returns")


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    print("CHAE — демо на данных интернет-магазина")
    print("Открытые (синтетические) данные за 12 недель\n")

    # 1. Создаём движок
    engine = CHAE(CHAEConfig(
        perception_backend="hash",
        seed_priors=False,          # свои priors важнее
        enable_continual_learning=True,
    ))

    # 2. Загружаем данные и засеиваем граф
    data = load_open_shop_data()
    seed_shop_causal_graph(engine)

    print(f"Загружено недель: {len(data)}")
    print(f"Узлов в графе: {engine.causal.graph.summary()['nodes']}")
    print(f"Рёбер в графе: {engine.causal.graph.summary()['edges']}")

    # 3. Регистрируем простой инструмент (можно заменить на реальный API)
    def shop_search(query: str = "") -> str:
        return f"[shop-search] Найдено по запросу «{query}»: 3 товара, средний рейтинг 4.6"

    engine.tools.register("shop_search", shop_search)

    # 4. Обычные вопросы «почему?»
    print_section("1. Причинные вопросы")

    questions = [
        "Почему растёт выручка?",
        "Что влияет на количество заказов?",
        "Как реклама связана с трафиком?",
        "Почему появляются возвраты?",
    ]

    for q in questions:
        result = engine.think(q)
        print(f"\nВопрос: {q}")
        print(f"  Уверенность : {result['causal_confidence']:.2f}")
        print(f"  Объяснение  : {result['explanation']}")
        if result["actions"]:
            print(f"  Действия    : {result['actions']}")

    # 5. Интервенции («что если?»)
    print_section("2. Интервенции")

    print("\n→ Что будет, если сильно увеличить рекламу (ads_spend = True)?")
    inter1 = engine.intervene("ads_spend", True)
    print(json.dumps(inter1, ensure_ascii=False, indent=2))

    print("\n→ Что будет, если поднять цену (price = True)?")
    inter2 = engine.intervene("price", True)
    print(json.dumps(inter2, ensure_ascii=False, indent=2))

    print("\n→ Что будет, если убрать высокий сезон (season_high = False)?")
    inter3 = engine.intervene("season_high", False)
    print(json.dumps(inter3, ensure_ascii=False, indent=2))

    # 6. Контрфактуалы
    print_section("3. Контрфактуалы")

    cf = engine.counterfactual("ads_spend", "no_ads")
    print("Сравнение: ads_spend vs no_ads")
    print(json.dumps(cf, ensure_ascii=False, indent=2))

    # 7. Статистика памяти после работы
    print_section("4. Память после работы")
    stats = engine.memory.stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # 8. Показать итоговый граф
    print_section("5. Итоговый причинный граф")
    summary = engine.causal.graph.summary()
    print(f"Узлы : {summary['nodes']}")
    print("Рёбра:")
    for edge in summary.get("edges", []):
        print(f"  {edge}")

    print("\nГотово. Можно открывать визуализацию:")
    print("  engine.visualize()  →  http://127.0.0.1:8050")


if __name__ == "__main__":
    main()