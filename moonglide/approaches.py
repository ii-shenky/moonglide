"""Параметры заходов по полосам: угол глиссады, высота над порогом, FAF и IF.

Без них — 3°, 15 м, сплошная глиссада на 18 км. Источник цифр — схемы захода (AIP страны). Общие — в
approaches.toml (в репозитории), свои правки — в approaches.local.toml (в .gitignore), его записи главнее:

    [XXXX."09"]
    angle = 3.0        # угол глиссады, °
    tch_ft = 50        # высота пролёта порога
    faf_nm = 5.0       # FAF — начало снижения, морских миль от порога
    if_nm = 10.0       # IF — начало прямого участка (до FAF — площадка на высоте FAF)
    note = "ILS RWY 09"
"""
import tomllib
from pathlib import Path

NM_KM, FT_M = 1.852, 0.3048


def load(*paths: Path | None) -> dict[tuple[str, str], dict]:
    """Записи из файлов по порядку; запись следующего файла заменяет запись предыдущего для той же полосы."""
    out = {}
    for path in paths:
        out.update(_load(path))
    return out


def _load(path: Path | None) -> dict[tuple[str, str], dict]:
    if not path or not path.exists():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    out = {}
    for airport, ends in data.items():
        for ident, v in ends.items():
            a = {"note": v.get("note", ""), "file": path.name}
            if "angle" in v:
                a["angle"] = float(v["angle"])
            if "tch_ft" in v:
                a["tch_m"] = float(v["tch_ft"]) * FT_M
            if "faf_nm" in v:
                a["faf_km"] = float(v["faf_nm"]) * NM_KM
            if "if_nm" in v:
                a["length_km"] = float(v["if_nm"]) * NM_KM
            out[(airport.upper(), str(ident).upper())] = a
    return out
