"""Веб-сервер moonglide: страница с картой и API.

    /                      страница (moonglide/www/index.html), /help — инструкция (help.html)
    /api/airport?code=LCLK название и концы полос
    /api/night?airport=LCLK&runway=22&date=2026-11-26&body=moon[&angle=3&length=18]
                           линии «где стоять» на сутки от местного полудня даты
    /api/point?…&lat=&lon= когда из этой точки глиссада идёт по диску
    /api/point_search?airport=LCLK&date=…&days=60&body=moon&lat=&lon=
                           ближайшие окна из точки по всем полосам аэропорта (60 дней × 2 полосы — ~4 с)
    /api/search?airport=LCLK&runway=22&date=2026-11-09&days=60&body=moon&size=fit
               [&min_alt=8&min_illum=0.3&sky=night,twilight,day]
                           удачные окна на days суток вперёд (size: fit — A320 целиком в диске 25–100 %,
                           big — до двух дисков, any — 10 % … 5 дисков); 60 суток считаются ~15 с, потом —
                           из кеша, фильтры применяются к моментам до склейки в окна

Тяжёлые расчёты: не больше WORKERS одновременно и PENDING разных в работе и очереди, одинаковые запросы
склеиваются (второй ждёт первого и получает его результат или ошибку), результаты — в LRU-кеше. Соединений — не
больше MAX_CONN. Ошибки — JSON {"error": …}: 400 — параметры, 503 — занято или нет рельефа.
"""
import gzip
import http.server
import json
import logging
import math
import re
import threading
import time
import urllib.parse
from collections import OrderedDict
from datetime import date as Date, datetime
from pathlib import Path

from . import plan
from .dem import ATTRIBUTION as DEM_ATTRIBUTION, DemError

log = logging.getLogger(__name__)
WWW = Path(__file__).parent / "www"
PAGES = {"/": ("index.html", "text/html; charset=utf-8"), "/help": ("help.html", "text/html; charset=utf-8")}
CODE_RE = re.compile(r"^[A-Za-z0-9-]{2,8}$")
IDENT_RE = re.compile(r"^[A-Za-z0-9-]{1,6}$")
# DE421: 1899-07-29 … 2053-10-09; с запасом на сутки и на поиск вперёд
EPH_FIRST, EPH_LAST = Date(1900, 1, 1), Date(2053, 10, 1)
WORKERS = 2  # тяжёлых расчётов одновременно
WAIT_S = 120.0
PENDING = 8  # разных тяжёлых расчётов в работе и в очереди
MAX_CONN = 64  # одновременных соединений; дальше — сразу 503


class BadRequest(Exception):
    pass


class Busy(Exception):
    pass


class _Job:
    def __init__(self):
        self.done = threading.Event()
        self.result = None
        self.error: BaseException | None = None


class Memo:
    """LRU-кеш расчётов. Одинаковые одновременные запросы ждут один расчёт и получают его результат — или его
    ошибку (без повтора). Разных расчётов в работе и в очереди — не больше pending (дальше сразу Busy), считаются
    одновременно не больше workers; ожидание — до общего срока wait от прихода запроса."""

    def __init__(self, size: int = 32, workers: int = WORKERS, wait: float = WAIT_S, pending: int = PENDING):
        self.lock = threading.Lock()
        self.data: OrderedDict = OrderedDict()
        self.inflight: dict = {}
        self.size, self.wait, self.pending = size, wait, pending
        self.sem = threading.BoundedSemaphore(workers)

    def get(self, key, fn):
        deadline = time.monotonic() + self.wait
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
            job = self.inflight.get(key)
            owner = job is None
            if owner:
                if len(self.inflight) >= self.pending:
                    raise Busy("сервер занят — повторите через минуту")
                job = self.inflight[key] = _Job()
        if not owner:
            if not job.done.wait(max(0.0, deadline - time.monotonic())):
                raise Busy("сервер занят — повторите через минуту")
            if job.error is not None:
                raise job.error
            return job.result
        try:
            if not self.sem.acquire(timeout=max(0.0, deadline - time.monotonic())):
                raise Busy("сервер занят — повторите через минуту")
            try:
                job.result = fn()
            finally:
                self.sem.release()
            with self.lock:
                self.data[key] = job.result
                while len(self.data) > self.size:
                    self.data.popitem(last=False)
            return job.result
        except BaseException as e:
            job.error = e
            raise
        finally:
            with self.lock:
                self.inflight.pop(key, None)
            job.done.set()


def _float(q, name, lo, hi, default=None):
    v = q.get(name)
    if v is None:
        if default is None:
            raise BadRequest(f"нужен параметр {name}")
        return default
    try:
        x = float(v)
    except ValueError:
        raise BadRequest(f"{name} — число")
    if not math.isfinite(x) or not lo <= x <= hi:
        raise BadRequest(f"{name}: от {lo} до {hi}")
    return x


def _int(q, name, lo, hi, default):
    try:
        x = int(q.get(name, default))
    except ValueError:
        raise BadRequest(f"{name} — целое число")
    if not lo <= x <= hi:
        raise BadRequest(f"{name}: от {lo} до {hi}")
    return x


def _date(q, days: int = 1) -> str:
    s = q.get("date", "")
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            raise ValueError
        d = datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise BadRequest("date — настоящая дата ГГГГ-ММ-ДД")
    if d < EPH_FIRST or d.toordinal() + days > EPH_LAST.toordinal():
        raise BadRequest(f"даты — с {EPH_FIRST} по {EPH_LAST} (эфемериды DE421), включая срок поиска")
    return d.isoformat()


class Handler(http.server.BaseHTTPRequestHandler):
    ctx = None  # sky, dem, runways, airports, approaches
    server_version = "moonglide"
    memo = Memo()
    SIZES = {"fit": (0.25, 1.0), "big": (0.25, 2.0), "any": (0.1, 5.0)}

    def do_GET(self):
        path, _, query = self.path.partition("?")
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(query).items()}
        try:
            if path in PAGES:
                name, ctype = PAGES[path]
                return self._send(200, (WWW / name).read_bytes(), ctype)
            api = {"/api/airport": self._airport, "/api/night": self._night, "/api/point": self._point,
                   "/api/point_search": self._point_search, "/api/search": self._search}.get(path)
            if api is None:
                return self._json({"error": "нет такой страницы"}, code=404)
            return self._json(api(q))
        except BadRequest as e:
            self._json({"error": str(e)}, code=400)
        except Busy as e:
            self._json({"error": str(e)}, code=503)
        except DemError as e:
            log.warning("%s", e)
            self._json({"error": "рельеф сейчас недоступен (Copernicus DEM) — повторите позже"}, code=503)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            log.exception("веб: %s", self.path)
            self._json({"error": "внутренняя ошибка сервера"}, code=500)

    # --- параметры
    def _airport(self, q):
        code = q.get("code", "")
        if not CODE_RE.match(code):
            raise BadRequest("код аэропорта — 2–8 латинских букв и цифр")
        ends = self.ctx.runways.of(code)
        if not ends:
            raise BadRequest(f"{code.upper()}: аэропорт не найден или у него нет полос с координатами")
        return {"code": code.upper(), "info": self.ctx.airports.get(code),
                "runways": [{"ident": e.ident, "heading": round(e.heading, 1), "lat": e.lat, "lon": e.lon,
                             "elev_m": round(e.elev_m, 1), "opposite": e.opposite, "length_m": round(e.length_m)}
                            for e in ends]}

    def _ends(self, q):
        code = q.get("airport", "")
        if not CODE_RE.match(code):
            raise BadRequest("код аэропорта — 2–8 латинских букв и цифр")
        ends = self.ctx.runways.of(code)
        if not ends:
            raise BadRequest(f"{code.upper()}: аэропорт не найден")
        return ends

    def _body(self, q) -> str:
        body = q.get("body", "moon")
        if body not in ("moon", "sun"):
            raise BadRequest("body — moon или sun")
        return body

    def _request(self, q, days: int = 1) -> plan.Request:
        ends = self._ends(q)
        ident = q.get("runway", "")
        if not IDENT_RE.match(ident):
            raise BadRequest("полоса — 1–6 латинских букв и цифр")
        rw = next((e for e in ends if e.ident.upper() == ident.upper()), None)
        if rw is None:
            raise BadRequest(f"у {ends[0].airport} нет полосы {ident} (есть {', '.join(e.ident for e in ends)})")
        return self._req_for(rw, _date(q, days), self._body(q), q)

    def _req_for(self, rw, date: str, body: str, q) -> plan.Request:
        """Параметры захода: из запроса (angle, length) > approaches.local.toml > 3°, 15 м, 18 км."""
        ap = self.ctx.approaches.get((rw.airport.upper(), rw.ident.upper()), {})
        angle = _float(q, "angle", 1.0, 6.0, ap.get("angle", 3.0))
        length = _float(q, "length", 2.0, 40.0, ap.get("length_km", 18.0))
        note = f"{ap['note']} ({ap['file']})" if ap else "типовая: 3°, 15 м над порогом, 18 км"
        return plan.Request(rw, date, body=body, angle=angle, length_km=length, tch_m=ap.get("tch_m", 15.0),
                            faf_km=ap.get("faf_km"), note=note)

    @staticmethod
    def _key(req: plan.Request) -> tuple:
        return (req.runway.airport, req.runway.ident, req.date, req.body, req.angle, req.length_km, req.tch_m,
                req.faf_km)

    def _latlon(self, q):
        return _float(q, "lat", -90.0, 90.0), _float(q, "lon", -180.0, 180.0)

    # --- расчёты
    def _night(self, q):
        req = self._request(q)

        def run():
            res = plan.night(self.ctx.sky, self.ctx.dem, req)
            res["attribution"] = {"dem": DEM_ATTRIBUTION}
            return res
        return self.memo.get(("night",) + self._key(req), run)

    def _point(self, q):
        req = self._request(q)
        lat, lon = self._latlon(q)
        return self.memo.get(("point",) + self._key(req) + (round(lat, 6), round(lon, 6)),
                             lambda: plan.point(self.ctx.sky, self.ctx.dem, req, lat, lon))

    def _point_search(self, q):
        ends = self._ends(q)
        days = _int(q, "days", 1, 120, 60)
        date, body = _date(q, days), self._body(q)
        lat, lon = self._latlon(q)
        reqs = [self._req_for(rw, date, body, {}) for rw in ends]
        key = ("near", ends[0].airport, date, body, days, round(lat, 6), round(lon, 6),
               tuple(self._key(r) for r in reqs))
        return self.memo.get(key, lambda: plan.point_search(self.ctx.sky, self.ctx.dem, reqs, days, lat, lon))

    def _search(self, q):
        days = _int(q, "days", 1, 120, 60)
        req = self._request(q, days)
        size = q.get("size", "fit")
        if size not in self.SIZES:
            raise BadRequest(f"size — {', '.join(self.SIZES)}")
        min_alt = _float(q, "min_alt", 0.0, 90.0, 0.0)
        min_illum = _float(q, "min_illum", 0.0, 1.0, 0.0)
        kinds = tuple(k for k in q.get("sky", ",".join(plan.SKY_KINDS)).split(",") if k)
        if not kinds or any(k not in plan.SKY_KINDS for k in kinds):
            raise BadRequest(f"sky — через запятую из {', '.join(plan.SKY_KINDS)}")
        lo, hi = self.SIZES[size]
        m = self.memo.get(("search",) + self._key(req) + (days, size),
                          lambda: plan.search_moments(self.ctx.sky, self.ctx.dem, req, days, lo, hi))
        return {"from": m["from"], "to": m["to"], "body": m["body"], "frac": m["frac"],
                "airport": req.runway.airport, "runway": req.runway.ident, "moments": len(m["moments"]),
                "windows": plan.group_windows(m, min_alt=min_alt, min_illum=min_illum, kinds=kinds)}

    # --- ответ
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
        self._send(code, body, "application/json; charset=utf-8")

    def _send(self, code: int, body: bytes, ctype: str):
        gz = len(body) > 2048 and "gzip" in (self.headers.get("Accept-Encoding") or "")
        if gz:
            body = gzip.compress(body, 5)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        log.debug("%s " + fmt, self.address_string(), *args)


class Server(http.server.ThreadingHTTPServer):
    """Не больше MAX_CONN соединений одновременно: лишним — сразу 503, без нового потока."""
    daemon_threads = True

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.slots = threading.BoundedSemaphore(MAX_CONN)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def serve(ctx, bind: str, port: int):
    Handler.ctx = ctx
    srv = Server((bind, port), Handler)
    srv.daemon_threads = True
    log.info("moonglide: http://%s:%d/", bind if bind not in ("0.0.0.0", "") else "localhost", port)
    srv.serve_forever()
