"""Сверка быстрого пути moonglide.sky (тело — точка в земных координатах минус наблюдатель, плюс рефракция)
с «честным» топоцентрическим расчётом Skyfield для каждого места отдельно.

    .venv/bin/python tools/check_sky.py

2026-11-09: Луна — до 0,5″, Солнце — до 0,3″ (200 случайных мест и моментов 2026–2028, высота до 2 км);
перевод unix-времени совпадает с from_datetime до миллисекунды.
"""
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from skyfield.api import wgs84

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moonglide import geo, sky  # noqa: E402

s = sky.Sky(Path(__file__).resolve().parent.parent / "cache")
random.seed(2)
worst = {"moon": 0.0, "sun": 0.0}
for _ in range(200):
    unix = 1767225600 + random.uniform(0, 3 * 365 * 86400)
    lat, lon, h = random.uniform(-60, 65), random.uniform(-180, 180), random.uniform(0, 2)
    t = s.ts.from_datetime(datetime.fromtimestamp(unix, timezone.utc))
    assert abs(t.tt - s.times(unix).tt) * 86400 < 1e-3, "перевод времени"
    for name in worst:
        ref = (s.earth + wgs84.latlon(lat, lon, elevation_m=h * 1000)).at(t).observe(s.eph[name]).apparent()
        ra, rz, _ = ref.altaz(temperature_C=sky.TEMPERATURE_C, pressure_mbar=sky.PRESSURE_MBAR)
        if ra.degrees < 0:
            continue
        _, alt, az, _ = sky.apparent_dir(s.body_ecef(name, unix), geo.to_ecef(lat, lon, h), lat, lon)
        a1, z1, a2, z2 = map(math.radians, (ra.degrees, rz.degrees, float(alt), float(az)))
        c = math.sin(a1) * math.sin(a2) + math.cos(a1) * math.cos(a2) * math.cos(z1 - z2)
        worst[name] = max(worst[name], math.degrees(math.acos(min(1.0, c))) * 3600)
print({k: f"до {v:.2f}″" for k, v in worst.items()}, "(радиус диска ~950″)")
