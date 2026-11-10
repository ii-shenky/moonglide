"""Луна и Солнце: Skyfield + эфемериды JPL DE421 (1900–2050, 17 МБ, скачиваются в кеш при первом запуске).

Тело отдаётся точкой в земных координатах (ITRS ≈ WGS84, км) на каждый момент — видимое из центра Земли
(со световым запаздыванием и аберрацией). Направление из любой точки на Земле = тело − точка: так параллакс
Луны (до 1°, два её диаметра) получается сам, без поправок. Суточная аберрация (~0,3″) не учтена.
Рефракция добавляется отдельно (`apparent_dir`), когда известна точка наблюдения.
"""
import math
from pathlib import Path

import numpy as np
from skyfield import almanac
from skyfield.api import Loader
from skyfield.earthlib import refract
from skyfield.framelib import itrs

from . import geo

RADIUS_KM = {"moon": 1737.4, "sun": 695700.0}
EPHEMERIS = "de421.bsp"
TEMPERATURE_C, PRESSURE_MBAR = 10.0, 1010.0  # для рефракции; погода на величину влияет на проценты


class Sky:
    def __init__(self, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        load = Loader(str(cache_dir), verbose=False)
        self.ts = load.timescale()
        self.eph = load(EPHEMERIS)
        self.earth = self.eph["earth"]

    def times(self, unix):
        """unix-время (число или массив) → время Skyfield. Без високосных секунд: unix их не считает,
        а ts.utc(1970, 1, 1, 0, 0, unix) — считает (разница 27 с ≈ 400″ для Луны)."""
        unix = np.asarray(unix, dtype=float)
        days = np.floor(unix / 86400)
        return self.ts.utc(1970, 1, 1 + days, 0, 0, unix - days * 86400)

    def body_ecef(self, name: str, unix) -> np.ndarray:
        """Центр тела в земных координатах, км: [3] или [T, 3]."""
        t = self.times(unix)
        app = self.earth.at(t).observe(self.eph[name]).apparent()
        return np.moveaxis(app.frame_xyz(itrs).km, 0, -1)

    def illumination(self, unix) -> np.ndarray:
        """Освещённая доля диска Луны 0…1."""
        return almanac.fraction_illuminated(self.eph, "moon", self.times(unix))

    def windows(self, name: str, lat: float, lon: float, h_km: float, start: float, end: float,
                min_alt: float = 5.0, step: float = 300.0) -> list[tuple[float, float]]:
        """Отрезки [начало, конец] (unix), когда тело выше min_alt (видимая высота) из точки lat, lon."""
        ts = np.arange(start, end + step, step)
        site = geo.to_ecef(lat, lon, h_km)

        def app_alt(t):
            return refract(geo.alt_az(self.body_ecef(name, t) - site, lat, lon)[0], TEMPERATURE_C, PRESSURE_MBAR)

        alt = app_alt(ts)
        # экстремумы высоты у порога: тело может подняться выше min_alt (или опуститься ниже) между отсчётами —
        # время экстремума (золотым сечением) добавляется в отсчёты, дальше переходы ищутся как обычно
        i = np.arange(1, len(ts) - 1)
        ext = i[((alt[i] >= alt[i - 1]) & (alt[i] >= alt[i + 1]) | (alt[i] <= alt[i - 1]) & (alt[i] <= alt[i + 1]))
                & (np.abs(alt[i] - min_alt) < 2.0)]
        if len(ext):
            sgn = np.where(alt[ext] >= alt[ext - 1], 1.0, -1.0)  # 1 — максимум, −1 — минимум
            a, b = ts[ext - 1].astype(float), ts[ext + 1].astype(float)
            g = (math.sqrt(5) - 1) / 2
            for _ in range(40):  # 600 с · 0,618⁴⁰ — доли миллисекунды
                c, d = b - g * (b - a), a + g * (b - a)
                fc, fd = sgn * app_alt(c), sgn * app_alt(d)
                a, b = np.where(fc > fd, a, c), np.where(fc > fd, d, b)
            ts = np.sort(np.concatenate([ts, (a + b) / 2]))
            alt = app_alt(ts)
        up = alt >= min_alt

        def edge(a, b, up_a):  # переход между a и b — делением пополам до 1 с
            while b - a > 1:
                m = (a + b) / 2
                if (app_alt(m) >= min_alt) == up_a:
                    a = m
                else:
                    b = m
            return (a + b) / 2

        out, cur = [], ts[0] if up[0] else None
        for i in range(1, len(ts)):
            if up[i] != up[i - 1]:
                x = edge(ts[i - 1], ts[i], bool(up[i - 1]))
                if up[i]:
                    cur = x
                else:
                    out.append((float(cur), float(x)))
                    cur = None
        if cur is not None:
            out.append((float(cur), float(ts[-1])))
        return out


def apparent_dir(body_ecef: np.ndarray, site_ecef: np.ndarray, lat, lon):
    """Видимое направление на тело из точки (с параллаксом и рефракцией).

    Возвращает (единичный вектор [..., 3], видимая высота °, азимут °, расстояние км).
    """
    v = body_ecef - site_ecef
    dist = np.linalg.norm(v, axis=-1)
    alt, az = geo.alt_az(v, lat, lon)
    alt_app = refract(alt, TEMPERATURE_C, PRESSURE_MBAR)
    e, n, u = geo.enu(lat, lon)
    a, z = np.radians(alt_app)[..., None], np.radians(az)[..., None]
    d = np.cos(a) * (np.sin(z) * e + np.cos(z) * n) + np.sin(a) * u
    return d, alt_app, az, dist


def radius_deg(name: str, dist_km) -> np.ndarray:
    return np.degrees(np.arcsin(RADIUS_KM[name] / np.asarray(dist_km)))
