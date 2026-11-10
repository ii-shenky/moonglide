"""Полосы из OurAirports (общественное достояние, https://ourairports.com/data/) и глиссада захода.

runways.csv скачивается в кеш при первом обращении (~4 МБ), в репозитории не хранится.
"""
import csv
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geo

RUNWAYS_URL = "https://davidmegginson.github.io/ourairports-data/runways.csv"
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
FT = 0.3048


@dataclass
class RunwayEnd:
    airport: str  # код ICAO (ident OurAirports)
    ident: str  # «22»
    lat: float  # порог (с учётом смещённого порога)
    lon: float
    elev_m: float
    heading: float  # истинный курс посадки на эту полосу, °
    length_m: float
    opposite: str  # «04»
    displaced_m: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.airport} {self.ident}"


def _download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "moonglide (https://github.com/ii-shenky/moonglide)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


class Runways:
    def __init__(self, cache_dir: Path):
        self.path = cache_dir / "runways.csv"
        self._by_airport: dict[str, list[RunwayEnd]] | None = None

    def _load(self):
        if self._by_airport is not None:
            return
        if not self.path.exists():
            _download(RUNWAYS_URL, self.path)
        out: dict[str, list[RunwayEnd]] = {}
        with self.path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("closed") == "1":
                    continue
                length = (_num(r["length_ft"]) or 0) * FT
                for me, other in (("le", "he"), ("he", "le")):
                    end = self._end(r, me, other, length)
                    if end:
                        out.setdefault(end.airport, []).append(end)
        self._by_airport = out

    @staticmethod
    def _end(r: dict, me: str, other: str, length: float) -> RunwayEnd | None:
        lat, lon = _num(r[f"{me}_latitude_deg"]), _num(r[f"{me}_longitude_deg"])
        if lat is None or lon is None:
            return None
        heading = _num(r[f"{me}_heading_degT"])
        olat, olon = _num(r[f"{other}_latitude_deg"]), _num(r[f"{other}_longitude_deg"])
        if olat is not None and olon is not None:  # курс по координатам концов точнее округлённого
            heading = float(geo.bearing_deg(lat, lon, olat, olon))
        if heading is None:
            return None
        elev = _num(r[f"{me}_elevation_ft"])
        if elev is None:
            elev = _num(r[f"{other}_elevation_ft"]) or 0.0
        disp = (_num(r[f"{me}_displaced_threshold_ft"]) or 0.0) * FT
        if disp:  # смещённый порог: посадка начинается дальше по полосе
            lat, lon = (float(v) for v in geo.destination(lat, lon, heading, disp / 1000))
        return RunwayEnd(airport=r["airport_ident"], ident=r[f"{me}_ident"], lat=lat, lon=lon, elev_m=elev * FT,
                         heading=heading, length_m=length, opposite=r[f"{other}_ident"], displaced_m=disp)

    def of(self, airport: str) -> list[RunwayEnd]:
        self._load()
        return self._by_airport.get(airport.upper(), [])

    def get(self, airport: str, ident: str) -> RunwayEnd:
        for end in self.of(airport):
            if end.ident.upper() == ident.upper():
                return end
        known = ", ".join(e.ident for e in self.of(airport)) or "нет полос"
        raise KeyError(f"{airport} {ident}: такой полосы нет ({known})")


@dataclass
class Glide:
    """Конечный участок захода: глиссада — прямая в пространстве от точки TCH над порогом назад по курсу
    под углом angle до FAF (faf_km от порога), дальше до IF (length_km) — горизонтальная площадка на высоте FAF.
    Без FAF — сплошная глиссада на всю длину. Параметры конкретных полос — approaches.py."""
    runway: RunwayEnd
    angle: float = 3.0  # °
    tch_m: float = 15.0  # высота пролёта порога
    length_km: float = 18.0  # до IF — начала прямого участка
    faf_km: float | None = None  # где начинается снижение по глиссаде
    step_km: float = 0.25
    speed_ms: float = 72.0  # скорость на финале (~140 узлов) — для оценки длительности пролёта

    def _axes(self):
        rw = self.runway
        e, n, u = geo.enu(rw.lat, rw.lon)
        back = math.radians(rw.heading + 180)
        horiz = math.sin(back) * e + math.cos(back) * n
        origin = geo.to_ecef(rw.lat, rw.lon, rw.elev_m / 1000)
        return origin, horiz, u

    def distances(self):
        return np.arange(0.0, self.length_km + 1e-9, self.step_km)

    def points(self):
        """(расстояние до порога км [M], точки в земных координатах км [M, 3], высота над порогом м [M])."""
        rw = self.runway
        d = self.distances()
        origin, horiz, u = self._axes()
        t = math.tan(math.radians(self.angle))
        line = origin + u * (self.tch_m / 1000) + d[:, None] * (horiz + u * t)  # прямая, не «над землёй»
        lat, lon, h = geo.from_ecef(line)
        if self.faf_km is not None and self.faf_km < self.length_km:
            _, _, h_faf = geo.from_ecef(origin + u * (self.tch_m / 1000) + self.faf_km * (horiz + u * t))
            h = np.where(d > self.faf_km, h_faf, h)  # площадка — постоянная высота
        pts = geo.to_ecef(lat, lon, h)
        return d, pts, (h * 1000 - rw.elev_m)

    def velocity_dirs(self):
        """Единичные векторы движения самолёта [M, 3] — касательная к самой линии захода (к порогу):
        на глиссаде — вдоль прямой вниз, на площадке — вдоль постоянной высоты над эллипсоидом."""
        _, pts, _ = self.points()
        if len(pts) < 2:
            _, horiz, u = self._axes()
            return geo.unit(-horiz - u * math.tan(math.radians(self.angle)))[None, :]
        return -geo.unit(np.gradient(pts, axis=0))


class Airports:
    """Названия аэропортов (airports.csv OurAirports, ~12 МБ в кеше): код → (название, город, IATA)."""

    def __init__(self, cache_dir: Path):
        self.path = cache_dir / "airports.csv"
        self._info: dict[str, dict] | None = None

    def get(self, ident: str) -> dict | None:
        if self._info is None:
            if not self.path.exists():
                _download(AIRPORTS_URL, self.path)
            with self.path.open(newline="", encoding="utf-8") as f:
                self._info = {r["ident"]: {"name": r["name"], "city": r["municipality"], "iata": r["iata_code"],
                                           "country": r["iso_country"]}
                              for r in csv.DictReader(f) if r["type"] in ("large_airport", "medium_airport", "small_airport")}
        return self._info.get(ident.upper())
