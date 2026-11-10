#!/usr/bin/env python3
"""Выгружает историю разработки moonglide с Claude Code в ai/sessions/ — для тех, кто продолжит проект.

    python3 tools/export_context.py --scrub .scrub

Берёт записи сессий Claude Code (~/.claude/projects/<папка проекта>/*.jsonl): папки moonglide и — с момента,
когда в ней начался moonglide (MOONGLIDE_START), — папки nebo, откуда проект вырос. Пишет ai/sessions/ГГГГ-ММ-ДД.md: реплики пользователя, ответы, короткие пометки о вызовах инструментов
(что делалось, без вывода), ответы пользователя на вопросы с вариантами, сводки при сжатии контекста.
Вывод команд, картинки и «мысли» не выгружаются.

Перед записью текст чистится: почта, токены, MAC- и IPv6-адреса заменяются заглушками, плюс замены
из файла --scrub (строки «регулярное выражение => замена»; файл локальный, в .gitignore — в нём то,
что прячем: адреса, домены, координаты). После выгрузки — проверить глазами и поиском.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = Path.home() / ".claude" / "projects"

BUILTIN_SCRUB = [
    (r"\beyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_-]+)*", "<токен>"),
    (r"\b\d+\+[A-Za-z0-9-]+@users\.noreply\.github\.com", "<id>+<аккаунт>@users.noreply.github.com"),
    (r"[A-Za-z0-9._%+-]+@(?!anthropic\.com|example\.com|users\.noreply\.github\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<почта>"),
    (r"\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b", "<mac>"),
    (r"\b(?:[0-9a-fA-F]{1,4}:){4,7}[0-9a-fA-F]{1,4}\b", "<ipv6>"),
    (r"\b[0-9a-fA-F]{32,}\b", "<hex>"),
    (r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "<uuid>"),
    (r"/(?:Users|home)/[^/\s]+/", "~/"),  # домашняя папка — без имени пользователя
]
MOONGLIDE_START = "2026-11-09T05:47:52"  # «Я хочу на основе nebo начать новый проект» — в сессии nebo
DROP_TAGS = re.compile(r"<(system-reminder|task-notification|local-command-stdout|local-command-stderr|local-command-caveat"
                       r"|command-message|command-args|ide_opened_file|ide_selection)>.*?</\1>", re.S)
COMMAND_NAME = re.compile(r"<command-name>(.*?)</command-name>", re.S)


def load_scrub(path: Path | None):
    rules = [(re.compile(p), r) for p, r in BUILTIN_SCRUB]
    if path:
        for line in path.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                pattern, _, repl = line.partition(" => ")
                rules.append((re.compile(pattern.strip()), repl.strip()))
    return rules


def scrub(text: str, rules) -> str:
    for rx, repl in rules:
        text = rx.sub(repl, text)
    return text


def user_text(content) -> str | None:
    """Текст пользователя без служебных вставок; None — сообщение целиком служебное."""
    if isinstance(content, str):
        parts = [content]
    else:
        parts = []
        for b in content:
            if b.get("type") == "text":
                parts.append(b["text"])
            elif b.get("type") == "image":
                parts.append("[картинка]")
    text = "\n".join(parts)
    text = COMMAND_NAME.sub(lambda m: m.group(1).strip(), text)
    text = DROP_TAGS.sub("", text).strip()
    return text or None


def tool_line(b) -> str:
    name, inp = b.get("name", "?"), b.get("input") or {}
    if name == "Bash":
        what = inp.get("description") or inp.get("command", "").splitlines()[0][:100]
    elif name in ("Read", "Write", "Edit", "NotebookEdit"):
        what = str(inp.get("file_path", "")).replace(str(ROOT) + "/", "")
    elif name == "WebSearch":
        what = inp.get("query", "")
    elif name == "WebFetch":
        what = inp.get("url", "")
    elif name == "AskUserQuestion":
        what = "; ".join(q.get("question", "") for q in inp.get("questions", []))
    else:
        what = inp.get("description") or inp.get("prompt", "")[:100] or ""
    return f"- `{name}` {what}".rstrip()


def export(transcripts: list[tuple[Path, str]], out: Path, rules):
    days = defaultdict(list)
    ask_ids = set()
    for path, since in transcripts:
        for line in path.open():
            d = json.loads(line)
            if since and (d.get("timestamp") or "") < since:
                continue
            att = d.get("attachment") if d.get("type") == "attachment" else None
            if isinstance(att, dict) and att.get("type") == "queued_command" and not d.get("isSidechain"):
                # сообщение, отправленное, пока агент работал (приходит ему посреди хода)
                ts = d.get("timestamp") or ""
                text = user_text(att.get("prompt") or "")
                if text:
                    days[ts[:10]].append(f"**Пользователь (пока агент работал)** · {ts[11:16]}\n\n{text}")
                continue
            if d.get("isSidechain") or d.get("isMeta") or d.get("type") not in ("user", "assistant"):
                continue
            msg, ts = d.get("message") or {}, d.get("timestamp") or ""
            day, hm = ts[:10], ts[11:16]
            content = msg.get("content")
            if d["type"] == "assistant":
                tools, texts = [], []
                for b in content if isinstance(content, list) else []:
                    if b.get("type") == "text" and b["text"].strip():
                        texts.append(b["text"].strip())
                    elif b.get("type") == "tool_use":
                        tools.append(tool_line(b))
                        if b.get("name") == "AskUserQuestion":
                            ask_ids.add(b.get("id"))
                if texts:
                    days[day].append(f"**Claude** · {hm}\n\n" + "\n\n".join(texts))
                if tools:
                    days[day].append("\n".join(tools))
                continue
            if isinstance(content, list) and any(b.get("type") == "tool_result" for b in content):
                for b in content:  # из результатов инструментов — только ответы пользователя на вопросы
                    if b.get("type") == "tool_result" and b.get("tool_use_id") in ask_ids:
                        c = b.get("content")
                        c = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [])
                        days[day].append(f"**Пользователь (ответ на вопрос)** · {hm}\n\n{c.strip()}")
                continue
            text = user_text(content)
            if not text:
                continue
            if text.startswith("This session is being continued from a previous conversation"):
                days[day].append(f"### Сводка контекста (автоматическое сжатие) · {hm}\n\n<details><summary>развернуть</summary>\n\n{text}\n\n</details>")
            else:
                days[day].append(f"**Пользователь** · {hm}\n\n{text}")

    sessions = out / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    for old in sessions.glob("*.md"):
        old.unlink()
    for day, items in sorted(days.items()):
        body = f"# {day}\n\n" + "\n\n---\n\n".join(items) + "\n"
        (sessions / f"{day}.md").write_text(scrub(body, rules))
        print(f"{sessions / (day + '.md')}: {len(items)} записей")


def project_dir(path: Path) -> Path:
    return PROJECTS / ("-" + str(path).strip("/").replace("/", "-"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scrub", type=Path, help="локальный файл замен «регулярное выражение => замена»")
    ap.add_argument("--out", type=Path, default=ROOT / "ai")
    args = ap.parse_args()
    sources = [(project_dir(ROOT.parent / "nebo"), MOONGLIDE_START), (project_dir(ROOT), "")]
    transcripts = []
    for folder, since in sources:
        for path in sorted(folder.glob("*.jsonl")):
            # из папки nebo — только сессии, где был moonglide
            if since and "moonglide" not in path.read_text():
                continue
            transcripts.append((path, since))
    export(transcripts, args.out, load_scrub(args.scrub))


if __name__ == "__main__":
    main()
