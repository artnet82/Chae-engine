# Причинный движок CHAE

[![Тесты](https://github.com/artnet82/Chae-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/artnet82/Chae-engine/actions/workflows/tests.yml)
[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)


**CHAE (Causal Hierarchical Agent Engine)** — это лёгкий Python-движок, который строит **явную причинно-следственную модель мира** и рассуждает по ней. В отличие от обёрток над LLM, ядро CHAE — это граф причинности, символьный планер и четырёхслойная иерархическая память. LLM здесь опционален и подключается как вспомогательный инструмент, а не как источник истины.

## Возможности

- **Причинное рассуждение** — ответы на вопросы «почему?» с цепочкой `причина → следствие (сила, доказательство)`
- **Интервенции** — «что если убрать дождь?» → `wet_ground = 0.0`
- **Контрфактуалы** — сравнение эффектов `fire` vs `no_fire`
- **Многошаговые пути** — `exercise → fitness → disease_risk → lifespan` (сила пути = произведение рёбер)
- **Иерархическая память** — рабочая, эпизодическая, семантическая, процедурная
- **Символьный планер** — детерминированные шаги `explain → request_evidence → call_tool → conclude`
- **Реестр инструментов** — регистрируйте любую функцию, планер вызовет её автоматически
- **Детерминизм** — MD5-хэш-эмбеддинги по умолчанию: 0 токенов, 0 GPU, 0 сетевых вызовов

## Установка

```bash
pip install -e .
# или со всеми зависимостями для разработки
pip install -e ".[dev]"
```

Быстрый старт

```python
from chae import CHAE, CHAEConfig

engine = CHAE(CHAEConfig(perception_backend="hash"))
engine.tools.register("поиск", lambda query="": f"результат для: {query}")

результат = engine.think("Почему дождь делает землю мокрой?")
print(результат["causal_confidence"])   # 0.9
print(результат["explanation"])         # "rain -> wet_ground (strength=0.90, evidence=prior)"
```

Архитектура

```
текст ──► Слой восприятия ──► Иерархическая память ──► Причинное ядро ──► Символьный планер ──► Инструменты
           (хэш | ST)         4 слоя                    │  ▲
                                                       │  └── Обнаружение причинности (PC, Granger)
                                                       ▼
                                                 Визуализация (Flask + vis.js)
```

Слои памяти

Слой Бэкенд Назначение
Рабочая deque(maxlen=N) Последние запросы и контекст
Эпизодическая список + эмбеддинги Поиск по косинусному сходству
Семантическая networkx.DiGraph Граф сущностей и связей
Процедурная collections.Counter Частоты цепочек действий (навыки)

Обнаружение причинности

Метод Функция Входные данные
PC-алгоритм discover_pc(data, alpha) Числовая матрица
Granger discover_granger(df, maxlag, alpha) Временные ряды

Визуализация

```python
from chae.visualization import create_app
create_app(engine.causal).run(port=8050)
# → http://127.0.0.1:8050
```

Лицензия

MIT — см. LICENSE.

Коммерческая версия

Pro-функции (нейронное обнаружение причинности, коннекторы к Prometheus / Datadog / Kubernetes, экспорт в Neo4j) распространяются отдельно как chae-pro. Ядро остаётся бесплатным и открытым навсегда.