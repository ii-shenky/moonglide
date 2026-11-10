"""План на сутки: линии «где стоять» по минутам, разбор выбранной точки, поиск удачных окон на недели вперёд.

Сутки — от местного солнечного полудня даты до следующего (по долготе, без часовых поясов):
так ночь не режется пополам в любой точке мира.
"""
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import geo
from .dem import Dem
from .runways import Glide, RunwayEnd
from .shadow import DT_BODY, EYE_M, clear_towards, crossing_seconds, lon_box, march, project
from .sky import Sky, apparent_dir, radius_deg

# типовые размахи крыла, м — для «самолёт займёт N% диска»
WINGSPANS = [("ATR 72", 27.1), ("A320 / B737", 35.8), ("B777", 64.8)]


@dataclass
class Request:
    runway: RunwayEnd
    date: str  # ГГГГ-ММ-ДД, местная дата начала ночи
    body: str = "moon"
    angle: float = 3.0
    length_km: float = 18.0  # до IF
    tch_m: float = 15.0
    faf_km: float | None = None  # None — сплошная глиссада до IF
    note: str = ""  # откуда параметры захода
    step_s: float = 120.0
    min_alt: float = 5.0
    glide: Glide = field(init=False)

    def __post_init__(self):
        self.glide = self.glide_at(0.25)

    def glide_at(self, step_km: float) -> Glide:
        return Glide(self.runway, angle=self.angle, tch_m=self.tch_m, length_km=self.length_km,
                     faf_km=self.faf_km, step_km=step_km)

    def span(self) -> tuple[float, float]:
        d = datetime.fromisoformat(self.date).replace(tzinfo=timezone.utc).timestamp()
        noon = d + 12 * 3600 - self.runway.lon / 15 * 3600
        return noon, noon + 86400


def _r(x, nd):
    return [round(float(v), nd) for v in x]


def night(sky: Sky, dem: Dem, req: Request) -> dict:
    rw, g = req.runway, req.glide
    start, end = req.span()
    win = sky.windows(req.body, rw.lat, rw.lon, rw.elev_m / 1000, start, end, min_alt=req.min_alt)
    dist, pts, height = g.points()
    glat, glon, _ = geo.from_ecef(pts)
    out = {
        "runway": {"airport": rw.airport, "ident": rw.ident, "lat": rw.lat, "lon": rw.lon, "elev_m": rw.elev_m,
                   "heading": rw.heading, "opposite": rw.opposite},
        "glide": {"angle": g.angle, "tch_m": g.tch_m, "length_km": g.length_km, "speed_ms": g.speed_ms,
                  "faf_km": g.faf_km, "note": req.note,
                  "dist_km": _r(dist, 2), "height_m": _r(height, 0), "lat": _r(glat, 6), "lon": _r(glon, 6)},
        "body": req.body, "span": [start, end], "windows": win, "lines": [],
    }
    if not win:
        return out
    ts = np.concatenate([np.arange(math.ceil(a / req.step_s) * req.step_s, b, req.step_s) for a, b in win])
    if len(ts) == 0:
        return out
    body = sky.body_ecef(req.body, ts)
    sh = project(pts, g.velocity_dirs(), g.speed_ms, req.body, body, dem, min_alt=req.min_alt,
                 body_next=sky.body_ecef(req.body, ts + DT_BODY))
    site = geo.to_ecef(rw.lat, rw.lon, rw.elev_m / 1000)
    # где Луна и Солнце в каждый момент (от порога) — для лучей на карте
    _, sun_alt, sun_az, _ = apparent_dir(sky.body_ecef("sun", ts), site, rw.lat, rw.lon)
    _, moon_alt, moon_az, _ = apparent_dir(sky.body_ecef("moon", ts), site, rw.lat, rw.lon)
    illum = sky.illumination(ts) if req.body == "moon" else np.ones(len(ts))
    for i, t in enumerate(ts):
        ok = sh.ok[i]
        if not ok.any():
            continue
        # куски подряд годных точек глиссады (гора между — разрыв линии)
        runs, cur = [], []
        for j in range(len(dist)):
            if ok[j]:
                cur.append(j)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        out["lines"].append({
            "t": float(t), "alt": round(float(sh.alt[i, 0]), 2), "az": round(float(sh.az[i, 0]), 2),
            "radius_deg": round(float(sh.radius_deg[i, 0]), 4), "illum": round(float(illum[i]), 3),
            "sun_alt": round(float(sun_alt[i]), 1),
            "sky": {"moon": [round(float(moon_alt[i]), 1), round(float(moon_az[i]), 1)],
                    "sun": [round(float(sun_alt[i]), 1), round(float(sun_az[i]), 1)]},
            # точки: [широта, долгота, полуширина м, до самолёта км, индекс точки глиссады, пролёт с, море 0/1]
            # (море — рельеф ровно 0: Copernicus так заполняет океан, у берега билинейно уже не 0)
            "runs": [[[round(float(sh.lat[i, j]), 6), round(float(sh.lon[i, j]), 6),
                       round(float(sh.half_width_m[i, j]), 1), round(float(sh.dist_km[i, j]), 3), j,
                       round(float(sh.crossing_s[i, j]), 2), int(sh.ground_m[i, j] == 0.0)] for j in run]
                     for run in runs],
        })
    return out


def _segment_angles(u, a, b):
    """Угол (рад) от направлений u [T, 3] до дуг между направлениями a [S, 3] и b [S, 3] (отрезки глиссады,
    как их видит наблюдатель) и где на отрезке ближайшая точка — доля 0…1 по углу (уточняется потом).

    Возвращает (угол [T, S], признак «внутри дуги» [T, S])."""
    n = np.cross(a, b)
    nn = np.linalg.norm(n, axis=-1)
    n = n / np.maximum(nn, 1e-15)[:, None]
    un = u @ n.T  # проекция на нормаль плоскости дуги
    na, bn = np.cross(n, a), np.cross(b, n)
    in_arc = (u @ na.T >= 0) & (u @ bn.T >= 0) & (u @ (a + b).T > 0) & (nn[None, :] > 1e-12)
    ang_plane = np.arcsin(np.clip(np.abs(un), 0, 1))
    ang_end = np.arccos(np.clip(np.maximum(u @ a.T, u @ b.T), -1, 1))
    return np.where(in_arc, ang_plane, ang_end), in_arc


def point(sky: Sky, dem: Dem, req: Request, lat: float, lon: float, start: float | None = None,
          end: float | None = None, step_s: float = 5.0, coarse_s: float = 60.0) -> dict:
    """Когда из точки (lat, lon) глиссада проходит по диску: окна, лучший момент, самолёт, тело.

    По умолчанию — сутки req.date; start/end — любой срок (поиск от точки на недели вперёд).
    Угол — до сплошных отрезков глиссады между точками через 50 м (вблизи глиссады соседние точки видны
    в градусах друг от друга). Два прохода: раз в coarse_s секунд с запасом 0,35° (тело движется относительно
    глиссады не быстрее ~0,25°/мин), затем раз в step_s секунд вокруг кандидатов.
    Самолёт должен быть виден (рельеф между нами, самолёт не «внутри холма»), а тело — не закрыто горами за ним;
    оба луча проверяются целиком, от самой камеры. Перекрытие тела — в каждый уточнённый момент: окно — только
    открытая часть. В каждом окне тела есть хотя бы один грубый отсчёт (короткие окна у порога высоты).
    """
    rw = req.runway
    if start is None:
        start, end = req.span()
    ground = float(dem.height_m(np.array([lat]), np.array([lon]))[0])
    here = geo.to_ecef(lat, lon, ground / 1000)
    eye = geo.to_ecef(lat, lon, (ground + EYE_M) / 1000)  # камера — как в project()
    fine = req.glide_at(0.05)
    dist, pts, height = fine.points()
    plat, plon, _ = geo.from_ecef(pts)
    to_p = pts - eye
    rng = np.linalg.norm(to_p, axis=-1)
    los = to_p / rng[:, None]
    # видна ли точка глиссады: луч от неё к нам не задевает рельеф и сама она не в рельефе
    _, blocked, inside = march(pts, los, 0.0, rng, dem)
    visible = ~(blocked | inside)
    res = {"lat": lat, "lon": lon, "ground_m": round(ground, 1), "sea": ground == 0.0, "windows": [],
           "visible_frac": float(visible.mean()), "runway": rw.ident}
    seg = np.nonzero(visible[:-1] & visible[1:])[0]  # отрезки с видимыми концами
    if len(seg) == 0:
        return res
    win = sky.windows(req.body, lat, lon, ground / 1000, start, end, min_alt=req.min_alt)
    if not win:
        return res
    a_dir, b_dir = los[seg], los[seg + 1]

    def angles(ts):
        body = sky.body_ecef(req.body, ts)
        u, alt, az, bdist = apparent_dir(body, eye[None, :], np.full(len(ts), lat), np.full(len(ts), lon))
        best_ang = np.empty(len(ts))
        best_seg = np.empty(len(ts), dtype=int)
        for c in range(0, len(ts), 2000):
            ang, _ = _segment_angles(u[c:c + 2000], a_dir, b_dir)
            k = np.argmin(ang, axis=1)
            best_seg[c:c + 2000] = k
            best_ang[c:c + 2000] = np.degrees(ang[np.arange(len(k)), k])
        return best_ang, best_seg, u, alt, az, radius_deg(req.body, bdist)

    coarse = np.unique(np.concatenate([np.arange(math.ceil(a / coarse_s) * coarse_s, b, coarse_s) for a, b in win]
                                      + [np.array([(a + b) / 2 for a, b in win])]))
    if len(coarse) == 0:
        return res
    cand = []
    for k in range(0, len(coarse), 20000):
        tc = coarse[k:k + 20000]
        ang, _, _, _, _, rad = angles(tc)
        cand.extend(tc[ang <= rad + 0.35].tolist())
    if not cand:
        return res
    offs = np.arange(-coarse_s, coarse_s + step_s / 2, step_s)
    ts = np.round((np.array(cand)[:, None] + offs[None, :]).ravel() / step_s) * step_s
    w_a, w_b = np.array([a for a, _ in win]), np.array([b for _, b in win])
    # отсчёты — внутри окон тела; короткое окно (меньше шага) получает свою середину
    ts = np.unique(np.concatenate([ts, (w_a + w_b) / 2]))
    wi = np.searchsorted(w_a, ts, side="right") - 1
    in_win = (wi >= 0) & (ts <= w_b[np.maximum(wi, 0)])
    ts = ts[in_win & (ts >= start) & (ts <= end)]
    if len(ts) == 0:
        return res
    ang, best_s, u, alt, az, rad = angles(ts)
    inside_disc = (ang <= rad) & (alt >= req.min_alt)
    # тело не закрыто горами — в каждый момент (а не только в лучший)
    top = dem.max_m(*lon_box(np.array([lat]), np.array([lon]), pad=0.3))
    k_in = np.nonzero(inside_disc)[0]
    if len(k_in):
        inside_disc[k_in] = clear_towards(np.repeat(eye[None, :], len(k_in), axis=0), u[k_in], dem, top)
    vdirs = fine.velocity_dirs() * (fine.speed_ms / 1000)
    w_end = w_b[np.searchsorted(w_a, ts, side="right") - 1]
    i = 0
    while i < len(ts):
        if not inside_disc[i]:
            i += 1
            continue
        k = i
        while k + 1 < len(ts) and inside_disc[k + 1] and ts[k + 1] - ts[k] <= step_s * 1.5:
            k += 1
        b = i + int(np.argmin(ang[i:k + 1]))
        sj = int(seg[best_s[b]])  # отрезок sj → sj+1
        # точка на отрезке, которую видно в направлении тела (проекция u на плоскость дуги)
        A, B = pts[sj], pts[sj + 1]
        n = np.cross(los[sj], los[sj + 1])
        n = n / max(np.linalg.norm(n), 1e-15)
        up = u[b] - (u[b] @ n) * n
        c1, c2 = np.cross(A - eye, up), np.cross(B - A, up)
        f = float(np.clip(-(c1 @ c2) / max(c2 @ c2, 1e-18), 0.0, 1.0))
        P = A + f * (B - A)
        lerp = lambda x: float(x[sj] + f * (x[sj + 1] - x[sj]))  # noqa: E731
        d_km = float(np.linalg.norm(P - eye))
        los_b = (P - eye) / d_km
        r = float(rad[b])
        u_next = apparent_dir(sky.body_ecef(req.body, ts[b] + DT_BODY), eye, lat, lon)[0]
        v = vdirs[sj] + f * (vdirs[sj + 1] - vdirs[sj])
        crossing = float(crossing_seconds(v[None, :], los_b[None, :], np.array([d_km]), u[b][None, :],
                                          u_next[None, :], np.array([r]))[0])
        sun_alt = float(apparent_dir(sky.body_ecef("sun", ts[b]), here, lat, lon)[1])
        res["windows"].append({
            "start": float(ts[i]), "end": float(min(ts[k] + step_s, end, w_end[k])), "best": float(ts[b]),
            "date": local_date(float(ts[b]), rw.lon), "runway": rw.ident,
            "miss_radii": round(float(ang[b] / r), 2),  # 0 — точно через центр, 1 — по краю
            "glide_km": round(lerp(dist), 2), "plane_height_m": round(lerp(height) + rw.elev_m, 0),
            "plane_dist_km": round(d_km, 2), "crossing_s": round(crossing, 2),
            "plane_lat": round(lerp(plat), 6), "plane_lon": round(lerp(plon), 6),
            "alt": round(float(alt[b]), 2), "az": round(float(az[b]), 2), "radius_deg": round(r, 4),
            "disc": [[name, round(span / (d_km * 1000) / (2 * math.radians(r)), 3)] for name, span in WINGSPANS],
            "illum": round(float(sky.illumination(ts[b])), 3) if req.body == "moon" else None,
            "sun_alt": round(sun_alt, 1), "sky": sky_kind(sun_alt),
        })
        i = k + 1
    return res


def point_search(sky: Sky, dem: Dem, reqs: list[Request], days: int, lat: float, lon: float) -> dict:
    """Ближайшие окна из точки на days суток вперёд — по всем концам полос (какая в ходу, решает ветер)."""
    out = {"lat": lat, "lon": lon, "days": days, "windows": [], "visible": {}}
    for req in reqs:
        rw = req.runway
        start, _ = req.span()
        r = point(sky, dem, req, lat, lon, start=start, end=start + days * 86400)
        out["ground_m"], out["sea"] = r["ground_m"], r["sea"]
        out["visible"][rw.ident] = round(r["visible_frac"], 2)
        out["windows"].extend(r["windows"])
    out["windows"].sort(key=lambda w: w["start"])
    return out


SKY_NIGHT, SKY_TWILIGHT = -12.0, 0.0  # Солнце ниже −12° — ночь, от −12° до 0° — сумерки, выше — день
SKY_KINDS = ("night", "twilight", "day")


def sky_kind(sun_alt: float) -> str:
    return "night" if sun_alt < SKY_NIGHT else "twilight" if sun_alt < SKY_TWILIGHT else "day"


def local_date(t: float, lon: float) -> str:
    """Дата суток (от местного солнечного полудня), в которые попадает момент t — как в Request.span()."""
    return datetime.fromtimestamp(t - 12 * 3600 + lon / 15 * 3600, timezone.utc).strftime("%Y-%m-%d")


def search_moments(sky: Sky, dem: Dem, req: Request, days: int, frac_min: float = 0.25, frac_max: float = 1.0,
                   step_s: float = 240.0, chunk: int = 300) -> dict:
    """Моменты на days суток вперёд, когда тень глиссады лежит не на море и самолёт (A320) занимает
    от frac_min до frac_max диаметра диска. Грубее, чем night(): глиссада через 500 м, время через 4 мин.
    Окна из них собирает group_windows() — уже с фильтрами (небо, высота, освещённость)."""
    runway, body = req.runway, req.body
    g = req.glide_at(0.5)
    dist, pts, height = g.points()
    start, _ = req.span()
    end = start + days * 86400
    win = sky.windows(body, runway.lat, runway.lon, runway.elev_m / 1000, start, end, min_alt=req.min_alt)
    out = {"from": start, "to": end, "body": body, "frac": [frac_min, frac_max], "step_s": step_s,
           "lon": runway.lon, "moments": []}
    if not win:
        return out
    ts = np.concatenate([np.arange(math.ceil(a / step_s) * step_s, b, step_s) for a, b in win])
    site = geo.to_ecef(runway.lat, runway.lon, runway.elev_m / 1000)
    span_m = dict(WINGSPANS)["A320 / B737"]
    for k in range(0, len(ts), chunk):
        tc = ts[k:k + chunk]
        sh = project(pts, g.velocity_dirs(), g.speed_ms, body, sky.body_ecef(body, tc), dem, min_alt=req.min_alt,
                     body_next=sky.body_ecef(body, tc + DT_BODY))
        frac = span_m / (np.maximum(sh.dist_km, 1e-6) * 1000) / (2 * np.radians(sh.radius_deg))
        good = sh.ok & (sh.ground_m != 0.0) & (frac >= frac_min) & (frac <= frac_max)
        sun_alt = apparent_dir(sky.body_ecef("sun", tc), site, runway.lat, runway.lon)[1]
        illum = sky.illumination(tc) if body == "moon" else np.ones(len(tc))
        for i, t in enumerate(tc):
            n = int(good[i].sum())
            if not n:
                continue
            js = np.nonzero(good[i])[0]
            j = int(js[np.argmin(np.abs(np.log(frac[i, js] / 0.6)))])
            out["moments"].append({
                "t": float(t), "n": n, "alt": round(float(sh.alt[i, j]), 1), "az": round(float(sh.az[i, j]), 1),
                "illum": round(float(illum[i]), 3), "sun_alt": round(float(sun_alt[i]), 1),
                "lat": round(float(sh.lat[i, j]), 6), "lon": round(float(sh.lon[i, j]), 6),
                "frac": round(float(frac[i, j]), 3), "plane_dist_km": round(float(sh.dist_km[i, j]), 2),
                "glide_km": round(float(dist[j]), 1), "plane_height_m": round(float(height[j] + runway.elev_m)),
                "crossing_s": round(float(sh.crossing_s[i, j]), 2), "ground_m": round(float(sh.ground_m[i, j])),
                "from_runway_km": round(float(geo.distance_km(runway.lat, runway.lon, sh.lat[i, j], sh.lon[i, j])), 1),
                "from_runway_bearing": round(float(geo.bearing_deg(runway.lat, runway.lon, sh.lat[i, j], sh.lon[i, j]))),
            })
    return out


def group_windows(m: dict, min_alt: float = 0.0, min_illum: float = 0.0, kinds=SKY_KINDS) -> list[dict]:
    """Окна из моментов search_moments — после фильтров: подряд идущие подходящие моменты с одинаковым небом.
    Фильтр применяется к каждому моменту до склейки, поэтому подходящая часть окна не теряется."""
    step = m["step_s"]
    out, cur = [], None
    for x in m["moments"]:
        kind = sky_kind(x["sun_alt"])
        if kind not in kinds or x["alt"] < min_alt or x["illum"] < min_illum:
            continue
        if cur and (kind != cur["sky"] or x["t"] - cur["end"] > step * 0.5):
            out.append(cur)
            cur = None
        if cur is None:
            cur = {"start": x["t"], "end": x["t"], "sky": kind, "points": 0, "best": None}
        cur["end"] = x["t"] + step
        cur["points"] += x["n"]
        if cur["best"] is None or x["n"] > cur["best"]["n"]:
            cur["best"] = dict(x)
    if cur:
        out.append(cur)
    for w in out:
        w["end"] = min(w["end"], m["to"])
        w["date"] = local_date(w["best"]["t"], m["lon"])
    return out


def search(sky: Sky, dem: Dem, req: Request, days: int, frac_min: float = 0.25, frac_max: float = 1.0, **flt) -> dict:
    """search_moments + group_windows одним вызовом (для скриптов)."""
    m = search_moments(sky, dem, req, days, frac_min, frac_max)
    return {"from": m["from"], "to": m["to"], "body": m["body"], "frac": m["frac"], "windows": group_windows(m, **flt)}
