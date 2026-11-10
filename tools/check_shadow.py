"""Независимая проверка shadow.project: из найденной точки на земле Skyfield сам считает видимое направление
на Луну (параллакс, рефракция) и на точку глиссады — они должны совпасть. Плюс сходимость итераций.

    .venv/bin/python tools/check_shadow.py [аэропорт полоса ГГГГ-ММ-ДД]
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from skyfield.api import wgs84

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moonglide import approaches, dem, geo, runways, shadow, sky  # noqa: E402

airport, ident, date = (sys.argv[1:4] if len(sys.argv) > 3 else ("LCLK", "22", "2026-11-26"))
C = Path(__file__).resolve().parent.parent / "cache"
S, D = sky.Sky(C), dem.Dem(C)
rw = runways.Runways(C).get(airport, ident)
root = Path(__file__).resolve().parent.parent
ap = approaches.load(root / "approaches.toml", root / "approaches.local.toml").get((airport, ident), {})
g = runways.Glide(rw, **{k: v for k, v in ap.items() if k in ("angle", "tch_m", "faf_km", "length_km")})
print("заход:", ap.get("note") or "типовой 3°")
d, pts, _ = g.points()
noon = datetime.fromisoformat(date).replace(hour=10, tzinfo=timezone.utc).timestamp()
win = S.windows("moon", rw.lat, rw.lon, 0, noon, noon + 86400, min_alt=5)
print(rw.name, date, "Луна выше 5°:", [(datetime.fromtimestamp(a, timezone.utc).strftime("%H:%M"),
                                         datetime.fromtimestamp(b, timezone.utc).strftime("%H:%M")) for a, b in win], "UTC")
ts = np.concatenate([np.arange(a, b, 600.0) for a, b in win])
body = S.body_ecef("moon", ts)


def sep(a1, z1, a2, z2):
    a1, z1, a2, z2 = map(math.radians, (a1, z1, a2, z2))
    return math.degrees(math.acos(min(1.0, math.sin(a1) * math.sin(a2) + math.cos(a1) * math.cos(a2) * math.cos(z1 - z2))))


worst, n = 0.0, 0
sh = shadow.project(pts, g.velocity_dirs(), g.speed_ms, "moon", body, D)
for i in range(0, len(ts), max(1, len(ts) // 12)):
    t = S.times(ts[i])
    for j in range(0, len(d), 12):
        if not sh.ok[i, j]:
            continue
        here = wgs84.latlon(sh.lat[i, j], sh.lon[i, j], elevation_m=sh.ground_m[i, j] + shadow.EYE_M)
        ma, mz, _ = (S.earth + here).at(t).observe(S.eph["moon"]).apparent().altaz(
            temperature_C=sky.TEMPERATURE_C, pressure_mbar=sky.PRESSURE_MBAR)
        plat, plon, ph = geo.from_ecef(pts[j])
        pa, pz, _ = (wgs84.latlon(float(plat), float(plon), elevation_m=float(ph) * 1000) - here).at(t).altaz()
        e = sep(ma.degrees, mz.degrees, pa.degrees, pz.degrees) * 3600
        worst, n = max(worst, e), n + 1
rad = float(np.median(sh.radius_deg)) * 3600
print(f"{n} точек: самолёт от центра Луны — до {worst:.1f}″ (радиус диска {rad:.0f}″)")

for it in (1, 2, 3, 4):
    s2 = shadow.project(pts, g.velocity_dirs(), g.speed_ms, "moon", body, D, iterations=it)
    m = sh.ok & s2.ok
    shift = geo.distance_km(sh.lat[m], sh.lon[m], s2.lat[m], s2.lon[m]) * 1000
    print(f"итераций {it}: отличие от 3 итераций — до {shift.max():.2f} м")
