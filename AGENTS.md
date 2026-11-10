# Для ИИ-агентов (Codex и других)

Сначала прочитать:

1. [CLAUDE.md](CLAUDE.md) — цель, решения, состояние проекта.
2. [ai/README.md](ai/README.md) — карта кода, грабли, на которые уже наступали, и как проверять изменения.

История разработки — [ai/sessions/](ai/sessions/) (диалог с Claude Code по дням).

Не читать: `CLAUDE.local.md`, `.scrub`, `approaches.local.toml` (личное и локальное, в репозитории их нет).

Проверка перед сдачей работы:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/check_sky.py
.venv/bin/python tools/check_shadow.py
```

Общение с пользователем — по-русски. Коммит и пуш — только по его просьбе.
