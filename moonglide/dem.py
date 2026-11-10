"""Рельеф: Copernicus DEM GLO-30 (30 м, по всему миру) с AWS Open Data, тайлы 1°×1° по запросу.

https://registry.opendata.aws/copernicus-dem/ — бесплатно для всех, с указанием источника (ATTRIBUTION ниже).
Тайлы (~5–10 МБ) кешируются на диске, в репозитории не хранятся.

Какие тайлы есть — по спискам tileList.txt обоих наборов (кешируются на 30 дней): нет в GLO-30 (часть стран
не выпущена) — берём GLO-90; нет и там — океан, высота 0. Ошибка загрузки — DemError, ничего не запоминается.

Это модель поверхности (DSM): леса и часть зданий в ней есть, отдельных домов и столбов — нет.
Высоты — над уровнем моря (геоид EGM2008); см. geo.py, почему их можно считать высотами над эллипсоидом.

Для проверки лучей на пересечение с рельефом у каждого тайла есть «максимумы блоков»: наибольшая высота в блоке
1/120° (~1 км), расширенная на соседние блоки — верхняя граница рельефа в пределах блока от любой точки блока
(shadow.march: грубый шаг по максимумам, точное пересечение — по клеткам сетки, grid_ppd).
"""
import logging
import math
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SETS = {  # набор → (адрес, префикс имени тайла)
    "30": ("https://copernicus-dem-30m.s3.amazonaws.com", "Copernicus_DSM_COG_10"),
    "90": ("https://copernicus-dem-90m.s3.amazonaws.com", "Copernicus_DSM_COG_30"),
}
# лицензия Copernicus DEM (бесплатно для всех): для производных данных — «produced using …» для каждого набора,
# которым пользуемся (GLO-30, а где его нет — GLO-90), и отказ от ответственности. Сами тайлы мы не раздаём.
ATTRIBUTION = ("produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH "
               "2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. "
               "Produced using Copernicus WorldDEM-90 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH "
               "2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. "
               "The organisations in charge of the Copernicus programme by law or by delegation do not incur "
               "any liability for any use of the Copernicus WorldDEM-30 and WorldDEM-90.")
MAX_TILES = 4  # в памяти: тайл 3600×3600 float32 — 52 МБ
BLOCKS_PER_DEG = 120  # блок максимумов — 1/120° (~925 м)
LIST_MAX_AGE = 30 * 86400
UA = "moonglide (https://github.com/ii-shenky/moonglide)"


class DemError(RuntimeError):
    """Рельеф недоступен (сеть, доступ) — расчёт нельзя делать «как будто море»."""


def tile_name(prefix: str, lat0: int, lon0: int) -> str:
    ns = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}"
    ew = f"{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"
    return f"{prefix}_{ns}_00_{ew}_00_DEM"


def norm_lon0(lo0: int) -> int:
    return (lo0 + 180) % 360 - 180


class Tile:
    def __init__(self, arr: np.ndarray, source: str):
        self.arr = arr
        self.source = source  # "30" | "90" | "ocean"
        rows, cols = arr.shape
        self.ppd_y, self.ppd_x = rows, cols  # точек на градус (PixelIsPoint: шаг 1/rows по широте)
        by, bx = max(1, rows // BLOCKS_PER_DEG), max(1, cols // BLOCKS_PER_DEG)
        nr, nc = math.ceil(rows / by), math.ceil(cols / bx)
        pad = np.full((nr * by, nc * bx), -1e9, dtype=np.float32)
        pad[:rows, :cols] = arr
        bm = pad.reshape(nr, by, nc, bx).max(axis=(1, 3))
        # +1 точка сетки за краем блока (билинейная интерполяция) и соседние блоки — расширение 3×3
        ext = np.pad(bm, 1, mode="edge")
        dil = np.max([ext[i:i + nr, j:j + nc] for i in range(3) for j in range(3)], axis=0)
        self.bmax, self.by, self.bx = dil, by, bx

    @classmethod
    def ocean(cls):
        return cls(np.zeros((BLOCKS_PER_DEG, BLOCKS_PER_DEG), dtype=np.float32), "ocean")


class Dem:
    def __init__(self, cache_dir: Path):
        self.dir = cache_dir / "dem"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._tiles: OrderedDict[tuple[int, int], Tile] = OrderedDict()
        self._lists: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    # --- загрузка
    def _get(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError) as e:
            raise DemError(f"рельеф: {url}: {e}") from e

    def _list(self, key: str) -> set[str]:
        if key not in self._lists:
            path = self.dir / f"tileList-{key}.txt"
            if not path.exists() or time.time() - path.stat().st_mtime > LIST_MAX_AGE:
                data = self._get(f"{SETS[key][0]}/tileList.txt")
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
            self._lists[key] = set(path.read_text().split())
        return self._lists[key]

    def _fetch(self, lat0: int, lon0: int) -> Tile:
        for key in ("30", "90"):
            base, prefix = SETS[key]
            name = tile_name(prefix, lat0, lon0)
            if name not in self._list(key):
                continue
            path = self.dir / f"{name}.tif"
            if not path.exists():
                log.info("рельеф: скачиваю %s", name)
                data = self._get(f"{base}/{name}/{name}.tif")
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
            import tifffile  # нужен imagecodecs (DEFLATE с предсказателем для float)
            with tifffile.TiffFile(path) as t:
                page = t.pages[0]
                _, _, _, x0, y0, _ = page.tags[33922].value
                arr = page.asarray().astype(np.float32)
            if abs(x0 - lon0) > 1e-6 or abs(y0 - (lat0 + 1)) > 1e-6:
                raise DemError(f"{name}: неожиданная привязка {x0}, {y0}")
            np.maximum(arr, -500, out=arr)  # пустые значения — в «море»
            return Tile(arr, key)
        return Tile.ocean()  # ни в одном наборе — океан

    def _tile(self, lat0: int, lon0: int) -> Tile:
        key = (lat0, norm_lon0(lon0))
        with self._lock:
            if key in self._tiles:
                self._tiles.move_to_end(key)
                return self._tiles[key]
            tile = self._fetch(*key)
            self._tiles[key] = tile
            while len(self._tiles) > MAX_TILES:
                self._tiles.popitem(last=False)
            return tile

    # --- запросы
    def _groups(self, lat, lon):
        lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
            raise ValueError("рельеф: координаты не числа")
        lon = (lon + 180) % 360 - 180
        lat0s, lon0s = np.floor(lat).astype(int), np.floor(lon).astype(int)
        keys = (lat0s + 90) * 360 + (lon0s + 180)
        for key in np.unique(keys).tolist():
            la0, lo0 = divmod(key, 360)
            m = keys == key
            yield self._tile(la0 - 90, lo0 - 180), la0 - 90, lo0 - 180, m, lat[m], lon[m]

    def height_m(self, lat, lon) -> np.ndarray:
        """Высота рельефа (м) в точках — билинейно по сетке."""
        out = np.zeros(np.shape(lat), dtype=float)
        for tile, la0, lo0, m, la, lo in self._groups(lat, lon):
            if tile.source == "ocean":
                continue
            arr = tile.arr
            rows, cols = arr.shape
            fy = (la0 + 1 - la) * tile.ppd_y
            fx = (lo - lo0) * tile.ppd_x
            i0 = np.clip(np.floor(fy).astype(int), 0, rows - 1)
            j0 = np.clip(np.floor(fx).astype(int), 0, cols - 1)
            i1, j1 = np.minimum(i0 + 1, rows - 1), np.minimum(j0 + 1, cols - 1)
            dy, dx = np.clip(fy - i0, 0, 1), np.clip(fx - j0, 0, 1)
            top = arr[i0, j0] * (1 - dx) + arr[i0, j1] * dx
            bot = arr[i1, j0] * (1 - dx) + arr[i1, j1] * dx
            out[m] = top * (1 - dy) + bot * dy
        return out

    def block_max(self, lat, lon) -> np.ndarray:
        """Верхняя граница рельефа в окрестности ~1 км вокруг точек (м)."""
        out = np.zeros(np.shape(lat), dtype=float)
        for tile, la0, lo0, m, la, lo in self._groups(lat, lon):
            nr, nc = tile.bmax.shape
            bi = np.clip(((la0 + 1 - la) * tile.ppd_y // tile.by).astype(int), 0, nr - 1)
            bj = np.clip(((lo - lo0) * tile.ppd_x // tile.bx).astype(int), 0, nc - 1)
            out[m] = tile.bmax[bi, bj]
        return out

    def grid_ppd(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Шаг сетки в точках: узлов на градус по широте и по долготе (линии сетки — кратные 1/ppd)."""
        py, px = np.zeros(np.shape(lat)), np.zeros(np.shape(lat))
        for tile, _, _, m, _, _ in self._groups(lat, lon):
            py[m], px[m] = tile.ppd_y, tile.ppd_x
        return py, px

    def is_sea(self, lat, lon) -> np.ndarray:
        """Море: рельеф ровно 0 (так Copernicus заполняет океан) или тайла нет. Озёра сюда не попадают."""
        return self.height_m(lat, lon) == 0.0

    def max_m(self, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> float:
        """Самая высокая точка тайлов, покрывающих прямоугольник. Долготы — «развёрнутые» (lon_min < lon_max,
        разница меньше 360): прямоугольник через ±180° задаётся, например, 179.5 … 180.5."""
        best = 0.0
        for la0 in range(max(-90, math.floor(lat_min)), min(89, math.floor(lat_max)) + 1):
            for lo0 in range(math.floor(lon_min), math.floor(lon_max) + 1):
                tile = self._tile(la0, lo0)
                best = max(best, float(tile.arr.max()))
        return best
