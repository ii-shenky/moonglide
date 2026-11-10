"""Сценарии из ревью Codex (2026-11-09 и 2026-11-10): каждый раньше давал неверный ответ.

    .venv/bin/python -m unittest discover -s tests -v

Рельеф — настоящие тайлы dem.Tile из синтетических массивов (GridDem: те же максимумы блоков и билинейная
интерполяция, что у Copernicus) или функция (FakeDem — только для ровной земли и грубых препятствий).
Пересечения сверяются с перебором шагом 0,5 м, независимо от shadow.march.
Нужен кеш (эфемериды, полосы) в cache/ — его создаёт первый запуск сервера.
"""
import math
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from moonglide import dem as D, geo, plan, runways, shadow, sky  # noqa: E402

CACHE = ROOT / "cache"
S = sky.Sky(CACHE)
R = runways.Runways(CACHE)


class GridDem(D.Dem):
    """Dem с тайлами из массивов: {(lat0, lon0): функция → np.ndarray}; остальное — океан."""

    def __init__(self, tiles):
        super().__init__(Path(tempfile.mkdtemp()))
        self.src = tiles

    def _fetch(self, lat0, lon0):
        f = self.src.get((lat0, lon0))
        return D.Tile(f().astype(np.float32), "30") if f else D.Tile.ocean()


class FakeDem:
    """Рельеф из функции; максимумы блоков — глобальный top (консервативно, но не проверяет блоки)."""

    def __init__(self, f, top=0.0):
        self.f, self.top = f, top

    def height_m(self, lat, lon):
        return self.f(np.asarray(lat, float), np.asarray(lon, float))

    def block_max(self, lat, lon):
        return np.full(np.shape(lat), self.top)

    def grid_ppd(self, lat, lon):
        return np.full(np.shape(lat), 3600.0), np.full(np.shape(lat), 3600.0)

    def max_m(self, *a):
        return self.top

    def is_sea(self, lat, lon):
        return self.height_m(lat, lon) == 0.0


def flat(lat, lon):
    return np.zeros(np.shape(lat))


def unix(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def typical(airport="LCLK", ident="22", date="2026-11-26"):
    return plan.Request(R.get(airport, ident), date)  # 3°, 15 м, 18 км, без площадки


def brute(p, u, length_km, dem, sign=1.0, lift=0.0, ds=0.0005):
    """Первая встреча перебором шагом ds (км) — None, если нет."""
    s = np.arange(0.0, length_km, ds)
    la, lo, h = geo.from_ecef(p + sign * s[:, None] * u)
    below = h * 1000 <= dem.height_m(la, lo) + lift
    return float(s[np.argmax(below)]) if below.any() else None


def ray(lat0, lon0, h0_km, lat1, lon1, h1_km):
    p = geo.to_ecef(lat0, lon0, h0_km)
    q = geo.to_ecef(lat1, lon1, h1_km)
    return p[None, :], geo.unit(q - p)[None, :], float(np.linalg.norm(q - p))


class Terrain(unittest.TestCase):
    """shadow.march — первая встреча луча с билинейной поверхностью, против перебора."""

    def check(self, p, u, length, dem, ds=0.0005):
        s, hit, below = shadow.march(p, u, 0.0, length, dem, sign=+1.0)
        b = brute(p[0], u[0], length, dem, ds=ds)
        self.assertIsNotNone(b, "сценарий без встречи — тест ничего не проверяет")
        self.assertTrue(hit[0], f"встреча пропущена (перебор: {b:.4f} км)")
        self.assertLessEqual(s[0], b + 1e-6)
        self.assertGreater(s[0], b - ds - 1e-6)

    def test_high_latitude_blocks(self):
        """№1 (2-е ревью): на 89° блок по долготе ~13 м — грубый шаг 250 м его перепрыгивал."""
        def t():
            a = np.zeros((3600, 360)); a[:, 180] = 400
            return a
        p, u, length = ray(89.2, 0.4, 0.2, 89.2, 0.6, 0.2)
        self.check(p, u, length, GridDem({(89, 0): t}))

    def test_tile_corner(self):
        """№1 (2-е ревью): луч коротко проходит через третий тайл у угла четырёх тайлов."""
        p, u, length = ray(34.9995, 33.9990, 0.5, 35.0010, 34.0012, 0.5)
        self.check(p, u, length, GridDem({(35, 34): lambda: np.full((3600, 3600), 3000.0)}))

    def test_shallow_peak(self):
        """№2 (2-е ревью): вершина между мелкими отсчётами — луч на 10 см ниже узла 100 м над фоном 99 м."""
        def t():
            a = np.full((3600, 3600), 99.0); a[1800, 1800] = 100
            return a
        la, lo = 35 - 1800 / 3600, 33 + 1800 / 3600
        p, u, length = ray(la, lo - 0.002, 0.0999, la, lo + 0.002, 0.0999)
        self.check(p, u, length, GridDem({(34, 33): t}), ds=0.0001)

    def test_narrow_ridge(self):
        """Хребет в одну клетку сетки (~30 м) поперёк луча на 5 км."""
        def t():
            a = np.zeros((3600, 3600)); a[:, 2000] = 2000
            return a
        lon_r = 33 + 2000 / 3600
        p, u, length = ray(34.5, lon_r - 0.05, 0.3, 34.5, lon_r + 0.05, 0.3)
        self.check(p, u, length, GridDem({(34, 33): t}))

    def test_random_vs_brute(self):
        """Неровный рельеф с редкими острыми пиками, 4 тайла, лучи во все стороны: ни одной пропущенной
        или поздней встречи, и каждая найденная — на поверхности."""
        def rough(seed):
            return lambda: np.random.default_rng(seed).random((3600, 3600)) ** 8 * 300
        dem = GridDem({(34, 33): rough(2), (35, 33): rough(3), (34, 34): rough(4), (35, 34): rough(5)})
        rng = np.random.default_rng(11)
        n = 60
        lat, lon = 34.99 + rng.random(n) * 0.02, 33.99 + rng.random(n) * 0.02
        az, el = rng.random(n) * 2 * np.pi, np.radians(rng.random(n) * 20 - 17)
        p = geo.to_ecef(lat, lon, rng.random(n) * 0.25)
        e, nn, up = geo.enu(lat, lon)
        u = np.cos(el)[:, None] * (np.sin(az)[:, None] * e + np.cos(az)[:, None] * nn) + np.sin(el)[:, None] * up
        s, hit, below = shadow.march(p, u, 0.0, 1.0, dem, sign=+1.0)
        checked = 0
        for i in range(n):
            if below[i]:
                continue
            b = brute(p[i], u[i], 1.0, dem)
            if b is None:
                if hit[i]:  # перебор мог не заметить касание — встреча должна быть на поверхности
                    la, lo, h = geo.from_ecef(p[i] + s[i] * u[i])
                    self.assertLess(abs(h * 1000 - dem.height_m(np.array([la]), np.array([lo]))[0]), 0.01)
                continue
            checked += 1
            self.assertTrue(hit[i], f"луч {i}: пропуск (перебор {b:.4f})")
            self.assertLessEqual(s[i], b + 1e-6, f"луч {i}")
            self.assertGreater(s[i], b - 0.0005 - 1e-6, f"луч {i}")
        self.assertGreater(checked, 20)


class Point(unittest.TestCase):
    def _plane_at(self, req, d_km):
        g = req.glide_at(0.001)  # точное положение (узлы через 1 м), не узел сетки разбора точки
        d, pts, h = g.points()
        j = int(np.argmin(abs(d - d_km)))
        la, lo, hh = geo.from_ecef(pts[j])
        return float(la), float(lo), float(hh) * 1000

    def _observer_for(self, req, d_km, t, dem=None):
        """Точка на земле, откуда самолёт на d_km видно точно на фоне центра Луны в момент t."""
        g = req.glide_at(0.001)
        d, pts, _ = g.points()
        j = int(np.argmin(abs(d - d_km)))
        sh = shadow.project(pts[j:j + 1], g.velocity_dirs()[j:j + 1], g.speed_ms, "moon",
                            S.body_ecef("moon", np.array([t])), dem or FakeDem(flat), min_alt=1.0)
        self.assertTrue(sh.ok[0, 0])
        return float(sh.lat[0, 0]), float(sh.lon[0, 0])

    def test_1_point_close_to_glide(self):
        """Вблизи глиссады соседние точки через 50 м видны в градусах друг от друга — угол до отрезков."""
        req = typical()
        t0 = unix("2026-11-26T20:00:00")
        r = plan.point(S, FakeDem(flat), req, 34.88851693, 33.64117673, start=t0 - 600, end=t0 + 600)
        self.assertTrue(r["windows"], "пролёт не найден")
        self.assertTrue(any(abs(w["glide_km"] - 1.03) < 0.1 for w in r["windows"]))

    def test_1b_projection_and_point_agree(self):
        """Точка из проекции и разбор этой точки дают тот же момент и центр диска — в том числе для самолёта
        посередине между узлами сетки разбора (через 50 м)."""
        req, t0 = typical(), unix("2026-11-26T20:00:00")
        for d_km in (0.6, 1.025, 3.0, 9.025):
            lat, lon = self._observer_for(req, d_km, t0)
            r = plan.point(S, FakeDem(flat), req, lat, lon, start=t0 - 900, end=t0 + 900)
            hit = [w for w in r["windows"] if w["start"] - 5 <= t0 <= w["end"] + 5]
            self.assertTrue(hit, f"{d_km} км: нет окна в {t0}")
            self.assertLess(hit[0]["miss_radii"], 0.15, f"{d_km} км")

    def test_2_plane_inside_hill(self):
        """Самолёт внутри холма невидим; наблюдатель — вне холма, на ровной земле."""
        req, t0 = typical(), unix("2026-11-26T20:00:00")
        lat, lon = self._observer_for(req, 1.0, t0)
        pla, plo, ph = self._plane_at(req, 1.0)
        hill = lambda la, lo: np.where(geo.distance_km(pla, plo, la, lo) < 0.05, 500.0, 0.0)  # noqa: E731
        self.assertEqual(float(hill(np.array([lat]), np.array([lon]))[0]), 0.0, "наблюдатель на холме")
        self.assertLess(ph, 500.0)
        r_flat = plan.point(S, FakeDem(flat), req, lat, lon, start=t0 - 300, end=t0 + 300)
        r_hill = plan.point(S, FakeDem(hill, top=500), req, lat, lon, start=t0 - 300, end=t0 + 300)
        self.assertTrue([w for w in r_flat["windows"] if abs(w["glide_km"] - 1.0) < 0.15])
        self.assertFalse([w for w in r_hill["windows"] if abs(w["glide_km"] - 1.0) < 0.15],
                         "самолёт внутри холма считается видимым")

    def test_3_mountain_behind_plane(self):
        req, t0 = typical(), unix("2026-11-26T20:00:00")
        lat, lon = self._observer_for(req, 5.0, t0)
        pla, plo, _ = self._plane_at(req, 5.0)
        az = float(geo.bearing_deg(lat, lon, pla, plo))
        bla, blo = geo.destination(pla, plo, az, 2.0)  # в 2 км за самолётом по лучу зрения
        mount = lambda la, lo: np.where(geo.distance_km(float(bla), float(blo), la, lo) < 0.5, 2000.0, 0.0)  # noqa: E731
        self.assertTrue(plan.point(S, FakeDem(flat), req, lat, lon, start=t0 - 120, end=t0 + 120)["windows"])
        r = plan.point(S, FakeDem(mount, top=2000), req, lat, lon, start=t0 - 120, end=t0 + 120)
        self.assertFalse(r["windows"], "Луна за горой, а окно есть")

    def test_near_camera_obstacle(self):
        """№3 (2-е ревью): препятствие 30 м в 3–14 м вокруг камеры — луч проверяется от самой камеры."""
        req, t0 = typical(), unix("2026-11-26T20:00:00")
        lat, lon = self._observer_for(req, 3.0, t0)
        ring = lambda la, lo: np.where(  # noqa: E731
            (geo.distance_km(lat, lon, la, lo) > 0.003) & (geo.distance_km(lat, lon, la, lo) < 0.014), 30.0, 0.0)
        self.assertTrue(plan.point(S, FakeDem(flat), req, lat, lon, start=t0 - 300, end=t0 + 300)["windows"])
        r = plan.point(S, FakeDem(ring, top=30), req, lat, lon, start=t0 - 300, end=t0 + 300)
        self.assertFalse(r["windows"], "стена вокруг камеры не закрывает обзор")

    def test_occlusion_each_moment(self):
        """№4 (2-е ревью): узкий высокий холм закрывает Луну в начале окна — эта часть не входит в окно."""
        req, t0 = typical(), unix("2026-11-26T20:00:00")
        lat, lon = self._observer_for(req, 9.0, t0)
        w0 = plan.point(S, FakeDem(flat), req, lat, lon, start=t0 - 1800, end=t0 + 1800)["windows"]
        self.assertTrue(w0)
        tm = w0[0]["start"] + 30
        eye = geo.to_ecef(lat, lon, shadow.EYE_M / 1000)
        _, alt, az, _ = sky.apparent_dir(S.body_ecef("moon", tm), eye, lat, lon)
        hla, hlo = geo.destination(lat, lon, float(az), 5.0)
        top = 5.0 * math.tan(math.radians(float(alt))) * 1000 + 300  # выше луча на Луну
        hill = lambda la, lo: np.where(geo.distance_km(float(hla), float(hlo), la, lo) < 0.03, top, 0.0)  # noqa: E731
        w = plan.point(S, FakeDem(hill, top=top), req, lat, lon, start=t0 - 1800, end=t0 + 1800)["windows"]
        self.assertFalse([x for x in w if x["start"] <= tm <= x["end"]], "закрытый момент внутри окна")
        self.assertTrue(w, "открытая часть окна потерялась")

    def test_8_window_bounds(self):
        req = typical()
        start, end = req.span()
        r = plan.point(S, FakeDem(flat), req, 34.88851693, 33.64117673)
        self.assertTrue(r["windows"], "сценарий без окон — тест ничего не проверяет")
        for w in r["windows"]:
            self.assertGreaterEqual(w["alt"], req.min_alt - 0.05)
            self.assertLessEqual(w["end"], end)

    def test_short_sky_window(self):
        """№5 (2-е ревью): Солнце едва выше 5° у кульминации — окно ~78 с между отсчётами раз в 300 с."""
        t0 = unix("2026-12-21T00:00:00")
        w = S.windows("sun", 61.72515147, 0.0, 0.0, t0, t0 + 86400, min_alt=5.0)
        self.assertEqual(len(w), 1)
        self.assertTrue(60 < w[0][1] - w[0][0] < 100, w)


class Search(unittest.TestCase):
    def test_5_filters_before_grouping(self):
        m = {"step_s": 240.0, "to": 1e12, "lon": 33.6, "moments": [
            {"t": 0.0, "n": 5, "alt": 5.5, "illum": 0.9, "sun_alt": -30, "lat": 0, "lon": 0},
            {"t": 240.0, "n": 4, "alt": 15.0, "illum": 0.9, "sun_alt": -30, "lat": 0, "lon": 0}]}
        ws = plan.group_windows(m, min_alt=8.0)
        self.assertEqual(len(ws), 1)
        self.assertEqual(ws[0]["best"]["alt"], 15.0)

    def test_11_antimeridian_box(self):
        _, _, lo1, lo2 = shadow.lon_box(np.array([0.0, 0.1]), np.array([179.95, -179.95]))
        self.assertLess(lo2 - lo1, 1.0)


class Server(unittest.TestCase):
    def test_13_memo_single_flight(self):
        """Два одновременных одинаковых запроса — один расчёт; одновременно — не больше workers."""
        from moonglide.web import Memo
        memo, calls, active, peak = Memo(workers=2), [], [0], [0]
        lock = threading.Lock()

        def slow(key):
            def run():
                with lock:
                    calls.append(key)
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.2)
                with lock:
                    active[0] -= 1
                return key
            return run
        threads = [threading.Thread(target=memo.get, args=(k, slow(k))) for k in ("a", "a", "a", "b", "c", "d")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(calls), ["a", "b", "c", "d"])
        self.assertLessEqual(peak[0], 2)

    def test_memo_shared_error(self):
        """№8 (2-е ревью): ошибка расчёта достаётся всем ожидающим, без повтора."""
        from moonglide.web import Memo
        memo, calls, errors = Memo(wait=5), [], []

        def boom():
            calls.append(1)
            time.sleep(0.05)
            raise RuntimeError("нет рельефа")

        def call():
            try:
                memo.get("k", boom)
            except RuntimeError as e:
                errors.append(str(e))
        threads = [threading.Thread(target=call) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(errors), 6)

    def test_memo_pending_limit(self):
        """№7 (2-е ревью): разных расчётов в работе и очереди — не больше pending, дальше сразу Busy."""
        from moonglide.web import Busy, Memo
        memo, gate = Memo(workers=1, pending=2, wait=5), threading.Event()
        threads = [threading.Thread(target=memo.get, args=(k, lambda: gate.wait(5))) for k in ("a", "b")]
        for t in threads:
            t.start()
        time.sleep(0.1)
        t_start = time.monotonic()
        with self.assertRaises(Busy):
            memo.get("c", lambda: 1)
        self.assertLess(time.monotonic() - t_start, 0.5)
        gate.set()
        for t in threads:
            t.join()
        self.assertEqual(memo.get("c", lambda: 1), 1)

    def test_date_format(self):
        """№9 (2-е ревью): дата без ведущих нулей — 400, а не 500 дальше в расчёте."""
        from moonglide.web import BadRequest, _date
        with self.assertRaises(BadRequest):
            _date({"date": "2026-1-1"})
        self.assertEqual(_date({"date": "2026-01-01"}), "2026-01-01")


if __name__ == "__main__":
    unittest.main()
