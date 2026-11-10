"""«Лунная тень» глиссады: где на земле стоять, чтобы глиссада проходила по диску Луны (или Солнца).

Для каждой точки глиссады P и момента t: луч из P против направления на тело до первой встречи с рельефом —
точка G. Направление берётся видимое из самой G (параллакс, рефракция), поэтому — итерации:
G₀ по направлению из P → направление из G₀ → G₁. Второй шаг сдвигает точку на метры, третий — меньше сантиметра
(tools/check_shadow.py: из найденных точек Skyfield сам видит самолёт в пределах нескольких угловых секунд
от центра Луны при радиусе ~1000″).

Полоса: стоя в G, видишь глиссаду через центр диска; отойдя вбок, — через край. Полуширина поперёк линии
в точке G: θ·D·√(1 + (n·r_h)²/sin²h), θ — угловой радиус диска, D — расстояние до самолёта, h — высота тела,
r_h — горизонтальная часть направления на самолёт, n — нормаль к линии на земле (опорная функция эллипса
«θ от направления на самолёт» на плоскости земли).
"""
import math
from dataclasses import dataclass

import numpy as np

from . import geo
from .dem import Dem
from .sky import apparent_dir, radius_deg

COARSE_KM = 0.25  # грубый шаг по лучу (не больше 0,9 блока максимумов — на высоких широтах меньше)
BLOCK_KM = 110.57 / 120  # блок максимумов (dem.BLOCKS_PER_DEG) по широте, км; по долготе — × cos(широты)
MAX_RAY_KM = 80.0
CHUNK = 4000  # лучей за раз (память)
SAMPLES = 1_600_000  # точек грубого прохода за раз
REFINE = 24  # делений пополам внутри куска с пересечением: 250 м / 2²⁴ — сотые доли миллиметра
DT_BODY = 10.0  # с — для угловой скорости тела
EYE_M = 1.7  # камера над землёй: точка «где стоять» — там, где луч опускается до этой высоты над рельефом


def _coarse_step(lat_max: float) -> float:
    """Грубый шаг, при котором любая точка отрезка — в соседнем блоке (±1) от одного из его концов."""
    return max(min(COARSE_KM, 0.9 * BLOCK_KM * math.cos(math.radians(min(lat_max, 89.99)))), 1e-3)


def _first_root(f0, fm, f1):
    """Отрезок, где f — квадратичная по τ ∈ [0, 1] (f0, fm, f1 — в 0, ½, 1). Первый τ с f ≤ 0, иначе nan."""
    A = 2 * f1 - 4 * fm + 2 * f0
    B = 4 * fm - f1 - 3 * f0
    C = f0
    with np.errstate(divide="ignore", invalid="ignore"):
        tv = np.where(A > 0, -B / (2 * A), np.nan)
        vmin = np.where((tv > 0) & (tv < 1), C - B * B / (4 * A), np.inf)
        hit = (f0 <= 0) | (f1 <= 0) | (vmin <= 0)
        disc = np.sqrt(np.maximum(B * B - 4 * A * C, 0.0))
        q = -0.5 * (B + np.where(B >= 0, disc, -disc))
        r1, r2 = q / A, C / q
        lin = -C / B  # A ≈ 0
        cands = np.stack([np.where(np.abs(A) > 1e-12, r1, lin), np.where(np.abs(A) > 1e-12, r2, lin)])
        cands = np.where((cands >= -1e-9) & (cands <= 1 + 1e-9), cands, np.inf)
        tau = np.clip(np.min(cands, axis=0), 0.0, 1.0)
    tau = np.where(f0 <= 0, 0.0, np.where(np.isfinite(np.min(cands, axis=0)), tau, np.where(f1 <= 0, 1.0, tau)))
    return np.where(hit, tau, np.nan)


def _exact(pc, uc, a, b, dem: Dem, sign: float, lift_m: float):
    """Первая встреча луча с билинейной поверхностью рельефа на отрезках [a, b] (по одному на строку).

    Между линиями сетки рельеф вдоль луча — квадратичная функция (билинейная от линейных широты и долготы),
    высота луча — линейная (кривизна на сотнях метров — миллиметры). Поэтому отрезок режется по линиям сетки,
    в каждом куске — 3 точки (концы и середина), и минимум «луч − рельеф» находится точно, как у параболы:
    касание вершины холма между отсчётами не теряется. Возвращает s (nan — встречи нет)."""
    n = len(a)
    if n == 0:
        return np.empty(0)
    pa = pc + sign * a[:, None] * uc
    pb = pc + sign * b[:, None] * uc
    la_a, lo_a, _ = geo.from_ecef(pa)
    la_b, lo_b, _ = geo.from_ecef(pb)
    lo_b = lo_a + (lo_b - lo_a + 180) % 360 - 180  # без скачка через ±180°
    ab = (b - a)[:, None]

    def at(tt, r=slice(None)):
        ss = a[r, None] + ab[r] * tt
        return geo.from_ecef(pc[r, None, :] + sign * ss[..., None] * uc[r, None, :])

    def f(tt, r=slice(None)):
        la, lo, h = at(tt, r)
        return h * 1000 - dem.height_m(la, lo) - lift_m

    # линии сетки, которые пересекает отрезок: широта и долгота в узлах — кратные 1/ppd (у тайла второго конца
    # сетка может быть другой — тогда и её линии)
    ts, tgt, axis = [np.zeros((n, 1)), np.ones((n, 1))], [np.full((n, 2), np.nan)], [np.zeros((n, 2), int)]
    py_a, px_a = dem.grid_ppd(la_a, lo_a)
    py_b, px_b = dem.grid_ppd(la_b, (lo_b + 180) % 360 - 180)
    for ax, ppd, va, vb in ((0, py_a, la_a, la_b), (1, px_a, lo_a, lo_b),
                            (0, np.where(py_b != py_a, py_b, 0.0), la_a, la_b),
                            (1, np.where(px_b != px_a, px_b, 0.0), lo_a, lo_b)):
        fa, fb = va * ppd, vb * ppd
        lo_i = np.floor(np.minimum(fa, fb))
        cnt = (np.floor(np.maximum(fa, fb)) - lo_i).astype(int)
        kmax = int(cnt.max())
        if kmax <= 0:
            continue
        k = lo_i[:, None] + 1 + np.arange(kmax)[None, :]
        target = k / np.where(ppd > 0, ppd, 1.0)[:, None]
        span = np.where(vb != va, vb - va, 1.0)
        valid = np.arange(kmax)[None, :] < cnt[:, None]
        ts.append(np.where(valid, np.clip((target - va[:, None]) / span[:, None], 0.0, 1.0), 1.0))
        tgt.append(np.where(valid, target, np.nan))
        axis.append(np.full(target.shape, ax))
    T, tgt, axis = np.concatenate(ts, axis=1), np.concatenate(tgt, axis=1), np.concatenate(axis, axis=1)
    if T.shape[1] > 2:
        # широта и долгота вдоль хорды не строго линейны (~1 см на 250 м) — шаг Ньютона по настоящим координатам,
        # иначе точка «на линии» у стыка тайлов окажется в соседнем тайле
        la, lo, _ = at(T)
        lo = lo_a[:, None] + (lo - lo_a[:, None] + 180) % 360 - 180
        val = np.where(axis == 0, la, lo)
        span = np.where(axis == 0, (la_b - la_a)[:, None], (lo_b - lo_a)[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(np.isfinite(tgt) & (span != 0), (tgt - val) / span, 0.0)
        T = np.clip(T + corr, 0.0, 1.0)
    T = np.sort(T, axis=1)
    M = (T[:, :-1] + T[:, 1:]) / 2

    # концы куска — изнутри него: на линии сетки у стыка тайлов значение берётся из соседнего тайла (разрыв)
    eps = 1e-7
    f0, f1, fM = f(np.clip(T[:, :-1] + eps, 0, 1)), f(np.clip(T[:, 1:] - eps, 0, 1)), f(M)
    tau = _first_root(f0, fM, f1)
    has = np.isfinite(tau)
    first = np.argmax(has, axis=1)
    rows = np.arange(n)
    t0, t1 = T[rows, first], T[rows, first + 1]
    est = t0 + tau[rows, first] * (t1 - t0)
    # уточнение по настоящей функции (страховка: парабола точна, пока внутри куска нет разрыва рельефа)
    sel = has.any(axis=1)
    out = np.where(sel, est, np.nan)
    r = np.nonzero(sel & (f0[rows, first] > 0))[0]
    if len(r):
        fe = f(est[r, None], r)[:, 0]
        r_ok = (fe <= 0) | (f1[r, first[r]] <= 0)
        r, fe = r[r_ok], fe[r_ok]
        lo_t, hi_t = np.where(fe <= 0, t0[r], est[r]), np.where(fe <= 0, est[r], t1[r])
        for _ in range(REFINE):
            m = (lo_t + hi_t) / 2
            under = f(m[:, None], r)[:, 0] <= 0
            hi_t, lo_t = np.where(under, m, hi_t), np.where(under, lo_t, m)
        out[r] = hi_t
    return a + (b - a) * out


def march(p, u, s_lo, s_hi, dem: Dem, sign: float = -1.0, lift_m: float = 0.0):
    """Первая встреча луча X(s) = p + sign·s·u с рельефом на [s_lo, s_hi]. p, u: [N, 3]; s_lo, s_hi: [N], км.

    Возвращает (s [N] — nan без встречи, hit [N], start_below [N]). start_below — начало луча уже в рельефе
    (самолёт «внутри холма»): это не встреча, а недопустимый луч. lift_m — «земля» выше рельефа на столько
    (высота камеры при поиске, где стоять).

    Грубый проход: отрезок пропускается, только если нижний его конец выше максимумов блоков обоих концов
    (шаг не больше 0,9 блока — любая точка отрезка в соседнем блоке от конца; расширение 3×3 это покрывает)
    и он не пересекает границу тайла (максимумы расширены только внутри тайла). Остальные отрезки — точно,
    по клеткам сетки (_exact): узкие хребты и вершины между отсчётами не теряются.
    """
    n = p.shape[0]
    s_lo = np.broadcast_to(np.asarray(s_lo, dtype=float), (n,)).copy()
    s_hi = np.broadcast_to(np.asarray(s_hi, dtype=float), (n,)).copy()
    s_hi = np.maximum(s_hi, s_lo)
    out = np.full(n, np.nan)
    if n == 0:
        return out, np.zeros(0, bool), np.zeros(0, bool)
    la0, lo0, h0 = geo.from_ecef(p + sign * s_lo[:, None] * u)
    start_below = h0 * 1000 <= dem.height_m(la0, lo0) + lift_m
    la1, _, _ = geo.from_ecef(p + sign * s_hi[:, None] * u)
    step = _coarse_step(float(max(np.max(np.abs(la0)), np.max(np.abs(la1)))) + 0.5)
    k = int(np.ceil(float(np.max(s_hi - s_lo)) / step)) + 1
    rows_per = max(1, min(CHUNK, SAMPLES // (k + 1)))
    for c0 in range(0, n, rows_per):
        idx = np.arange(c0, min(n, c0 + rows_per))
        pc, uc, lo_c, hi_c = p[idx], u[idx], s_lo[idx], s_hi[idx]
        kc = int(np.ceil(float(np.max(hi_c - lo_c)) / step)) + 1
        sc = np.minimum(lo_c[:, None] + np.arange(kc + 1)[None, :] * step, hi_c[:, None])
        la, lo, h = geo.from_ecef(pc[:, None, :] + sign * sc[..., None] * uc[:, None, :])
        bm = dem.block_max(la, lo) + lift_m + 1.0  # +1 м — прогиб хорды и округления
        hm = h * 1000
        low = np.minimum(hm[:, :-1], hm[:, 1:]) <= np.maximum(bm[:, :-1], bm[:, 1:])
        lon_n = (lo + 180) % 360 - 180
        tile_cross = (np.floor(la[:, :-1]) != np.floor(la[:, 1:])) | (np.floor(lon_n[:, :-1]) != np.floor(lon_n[:, 1:]))
        cand = (low | tile_cross) & (sc[:, 1:] > sc[:, :-1])
        ri, ki = np.nonzero(cand)  # по лучам, внутри луча — по порядку
        if len(ri) == 0:
            continue
        # кандидаты — по порядку вдоль луча, порциями 1, 4, 16, …: после первой встречи остальные (обычно уже
        # под землёй) не нужны
        uniq, start, cnt = np.unique(ri, return_index=True, return_counts=True)
        rank = np.arange(len(ri)) - np.repeat(start, cnt)
        done = np.zeros(len(pc), bool)
        lo_r, width = 0, 1
        while lo_r < int(cnt.max()):
            sel = (rank >= lo_r) & (rank < lo_r + width) & ~done[ri]
            if sel.any():
                r, kk = ri[sel], ki[sel]
                sx = np.concatenate([_exact(pc[r[i:i + CHUNK]], uc[r[i:i + CHUNK]], sc[r[i:i + CHUNK], kk[i:i + CHUNK]],
                                            sc[r[i:i + CHUNK], kk[i:i + CHUNK] + 1], dem, sign, lift_m)
                                     for i in range(0, len(r), CHUNK)])
                ok = np.isfinite(sx)
                rr, ss_ = r[ok], sx[ok]
                if len(rr):
                    first_r, first_i = np.unique(rr, return_index=True)  # первая встреча (порядок сохранён)
                    out[idx[first_r]] = ss_[first_i]
                    done[first_r] = True
            lo_r += width
            width *= 4
    hit = np.isfinite(out) & ~start_below
    out[~hit] = np.nan
    return out, hit, start_below


def lon_box(lat, lon, pad: float = 0.1):
    """Прямоугольник вокруг точек с «развёрнутыми» долготами (через ±180° не охватывает весь мир)."""
    ref = float(np.ravel(lon)[0])
    rel = (np.asarray(lon) - ref + 180) % 360 - 180
    return (float(np.min(lat)) - pad, float(np.max(lat)) + pad,
            ref + float(rel.min()) - pad, ref + float(rel.max()) + pad)


def clear_towards(p, u, dem: Dem, top_m: float, s_lo: float = 0.0):
    """Луч из p к телу (по u) не задевает рельеф, пока не поднимется выше top_m. p, u: [N, 3]."""
    lat, lon, h = geo.from_ecef(p)
    alt = np.radians(geo.alt_az(u, lat, lon)[0])
    need = np.clip((top_m / 1000 - h) / np.maximum(np.sin(alt), 0.01), s_lo, MAX_RAY_KM)
    _, blocked, start_below = march(p, u, np.full(len(p), s_lo), need, dem, sign=+1.0)
    return ~(blocked | start_below)


@dataclass
class Shadow:
    lat: np.ndarray  # [T, M] точка на земле
    lon: np.ndarray
    ground_m: np.ndarray  # высота рельефа там
    dist_km: np.ndarray  # до самолёта (точки глиссады)
    alt: np.ndarray  # видимая высота тела из точки, °
    az: np.ndarray
    radius_deg: np.ndarray  # угловой радиус диска
    half_width_m: np.ndarray  # полуширина полосы поперёк линии
    crossing_s: np.ndarray  # сколько секунд самолёт идёт по диску через центр
    ok: np.ndarray  # точка есть: луч дошёл до земли, тело выше min_alt, обзор не закрыт


def crossing_seconds(v, los, dist_km, u, u_next, rad_deg, dt=DT_BODY):
    """Сколько длится пролёт через центр диска: 2θ / |ω самолёта − ω тела| (угловые скорости на небе)."""
    v_perp = v - geo.dot(v, los)[..., None] * los
    w_plane = v_perp / np.maximum(dist_km, 1e-6)[..., None]  # рад/с, вектор поперёк луча зрения
    w_body = (u_next - u) / dt
    w_rel = np.linalg.norm(w_plane - w_body, axis=-1)
    return 2 * np.radians(rad_deg) / np.maximum(w_rel, 1e-9)


def project(glide_pts, velocity_dir, speed_ms: float, body_name: str, body_ecef, dem: Dem,
            min_alt: float = 3.0, iterations: int = 2, body_next=None) -> Shadow:
    """glide_pts: [M, 3], body_ecef: [T, 3] (body_next — то же через DT_BODY с) → Shadow с массивами [T, M]."""
    t_n, m_n = body_ecef.shape[0], glide_pts.shape[0]
    p = np.broadcast_to(glide_pts[None, :, :], (t_n, m_n, 3)).reshape(-1, 3)
    b = np.broadcast_to(body_ecef[:, None, :], (t_n, m_n, 3)).reshape(-1, 3)
    plat, plon, ph = geo.from_ecef(p)
    u, alt, _, _ = apparent_dir(b, p, plat, plon)
    sin_alt = np.sin(np.radians(np.maximum(alt, 0.5)))
    s_hi = np.minimum((ph + 0.5) / sin_alt * 1.2 + 0.5, MAX_RAY_KM)
    s, hit, _ = march(p, u, 0.0, s_hi, dem, lift_m=EYE_M)
    s = np.where(hit, s, 0.0)
    for _ in range(iterations - 1):
        g = p - s[:, None] * u
        glat, glon, _ = geo.from_ecef(g)
        u, alt, _, _ = apparent_dir(b, g, glat, glon)
        s2, hit2, _ = march(p, u, np.maximum(s - 0.3, 0.0), s + 0.3, dem, lift_m=EYE_M)
        s = np.where(hit & hit2, s2, s)
    g = p - s[:, None] * u  # камера (EYE_M над рельефом)
    glat, glon, _ = geo.from_ecef(g)
    ground = dem.height_m(glat, glon)  # рельеф в точке (у моря — ровно 0, см. dem.is_sea)
    u, alt, az, dist_body = apparent_dir(b, g, glat, glon)
    rad = radius_deg(body_name, dist_body)
    top = dem.max_m(*lon_box(glat, glon))
    ok = hit & (alt >= min_alt)
    ok[ok] = clear_towards(p[ok], u[ok], dem, top)

    # полуширина: нормаль к линии на земле — по соседним точкам той же минуты
    e, n_, up = geo.enu(glat, glon)
    los = geo.unit(p - g)  # на самолёт
    rh = np.stack([geo.dot(los, e), geo.dot(los, n_)], axis=-1)  # горизонтальная часть, длина cos h
    sin_h = np.maximum(geo.dot(los, up), 1e-3)
    gg = g.reshape(t_n, m_n, 3)
    tang = np.gradient(gg, axis=1) if m_n > 1 else np.zeros_like(gg)
    tang = tang.reshape(-1, 3)
    te, tn = geo.dot(tang, e), geo.dot(tang, n_)
    tl = np.hypot(te, tn)
    nrm = np.stack([-tn, te], axis=-1) / np.maximum(tl, 1e-12)[:, None]
    d = s
    half = np.radians(rad) * d * 1000 * np.sqrt(1 + (np.sum(nrm * rh, axis=-1) ** 2) / sin_h ** 2)

    # длительность пролёта по диску — с учётом движения самого тела
    vel = np.asarray(velocity_dir)
    if vel.ndim == 1:  # одно направление на всю глиссаду
        vel = np.broadcast_to(vel[None, :], (m_n, 3))
    v = np.broadcast_to(vel[None, :, :], (t_n, m_n, 3)).reshape(-1, 3) * (speed_ms / 1000)
    if body_next is not None:
        bn = np.broadcast_to(body_next[:, None, :], (t_n, m_n, 3)).reshape(-1, 3)
        u_next = apparent_dir(bn, g, glat, glon)[0]
    else:
        u_next = u
    crossing = crossing_seconds(v, los, d, u, u_next, rad)

    shape = (t_n, m_n)
    return Shadow(lat=glat.reshape(shape), lon=glon.reshape(shape), ground_m=ground.reshape(shape),
                  dist_km=d.reshape(shape), alt=alt.reshape(shape), az=az.reshape(shape),
                  radius_deg=rad.reshape(shape), half_width_m=half.reshape(shape),
                  crossing_s=crossing.reshape(shape), ok=ok.reshape(shape))
