"""Геодезия на numpy: WGS84, земные координаты (ECEF, км) ⇄ широта/долгота/высота, локальные оси.

Высоты в moonglide — над уровнем моря (так даны пороги в OurAirports и рельеф Copernicus), но считаются
как высоты над эллипсоидом. Ошибка — сдвиг всей картины по вертикали на высоту геоида (на Кипре ~25–30 м,
одинаково для глиссады и рельефа), на направления почти не влияет. Геоид поэтому не нужен.
"""
import numpy as np

A = 6378.137  # км
F = 1 / 298.257223563
E2 = F * (2 - F)
R_MEAN = 6371.0088


def to_ecef(lat, lon, h_km):
    """Широта, долгота (°), высота (км) → массив [..., 3] в км."""
    la, lo = np.radians(lat), np.radians(lon)
    s = np.sin(la)
    n = A / np.sqrt(1 - E2 * s * s)
    return np.stack([(n + h_km) * np.cos(la) * np.cos(lo), (n + h_km) * np.cos(la) * np.sin(lo),
                     (n * (1 - E2) + h_km) * s], axis=-1)


def from_ecef(xyz):
    """[..., 3] км → (широта °, долгота °, высота км). Боуринг, 3 итерации — доли миллиметра."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(3):
        s = np.sin(lat)
        n = A / np.sqrt(1 - E2 * s * s)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - E2 * n / (n + h)))
    s = np.sin(lat)
    n = A / np.sqrt(1 - E2 * s * s)
    return np.degrees(lat), np.degrees(lon), p / np.cos(lat) - n


def enu(lat, lon):
    """Единичные векторы восток, север, вверх (каждый [..., 3])."""
    la, lo = np.radians(lat), np.radians(lon)
    sl, cl, so, co = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    east = np.stack([-so, co, np.zeros_like(so)], axis=-1)
    north = np.stack([-sl * co, -sl * so, cl], axis=-1)
    up = np.stack([cl * co, cl * so, sl], axis=-1)
    return east, north, up


def dot(a, b):
    return np.sum(a * b, axis=-1)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def alt_az(vec, lat, lon):
    """Направление (земной вектор) → (высота над горизонтом °, азимут °) в точке lat, lon."""
    e, n, u = enu(lat, lon)
    ve, vn, vu = dot(vec, e), dot(vec, n), dot(vec, u)
    return np.degrees(np.arctan2(vu, np.hypot(ve, vn))), np.degrees(np.arctan2(ve, vn)) % 360


def destination(lat, lon, bearing_deg, dist_km):
    """Точка на сфере в dist_km от (lat, lon) по азимуту bearing_deg — для коротких расстояний хватает сферы."""
    la, lo, b = np.radians(lat), np.radians(lon), np.radians(bearing_deg)
    d = dist_km / R_MEAN
    la2 = np.arcsin(np.sin(la) * np.cos(d) + np.cos(la) * np.sin(d) * np.cos(b))
    lo2 = lo + np.arctan2(np.sin(b) * np.sin(d) * np.cos(la), np.cos(d) - np.sin(la) * np.sin(la2))
    return np.degrees(la2), (np.degrees(lo2) + 540) % 360 - 180


def distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_MEAN * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(x, y)) % 360
