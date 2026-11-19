"""
sdm_topoclim.py  (v2 : traitement par blocs + parallélisme)
============================================================
Covariables topo-climatiques pour un deepSDM, à partir de CHELSA (30") et d'un MNT 30 m.

Fonctions principales
---------------------
reproject_dem()               MNT géographique -> CRS métrique 30 m (streaming GDAL)
downscale_temperature()       un raster CHELSA de température -> grille du MNT
downscale_temperature_many()  plusieurs rasters d'un coup (le MNT n'est lu qu'une fois)
downscale_bioclim()           CHELSA mensuel brut -> bio1..bio11 à 30 m en UNE passe,
                              sans fichiers intermédiaires (recommandé)
compute_bioclim()             bio1..bio11 à partir de rasters mensuels déjà downscalés
compute_terrain()             altitude, pente, orientation, northness/eastness, TPI multi-rayons

Principe de fonctionnement
--------------------------
* La grille du MNT est découpée en blocs (block_size x block_size, multiple de 256).
* Chaque bloc est traité par un processus travailleur ; le processus principal est
  le seul à écrire (les GeoTIFF compressés ne supportent pas l'écriture concurrente).
* Downscaling : les données grossières (CHELSA + altitude moyenne par cellule) sont
  petites ; elles sont calculées une fois, stockées en .npy (mmap) et partagées
  entre travailleurs. Chaque bloc n'en lit que la zone utile.
* Pente / TPI : chaque bloc est lu avec une marge (halo) >= plus grand rayon de TPI,
  puis rogné ; le résultat est identique à un traitement en un seul bloc.
* La mémoire par travailleur est de l'ordre de quelques centaines de Mo pour
  block_size=512 ; total ~ n_workers x cette valeur.

IMPORTANT : sous Windows/macOS (et ici, par choix, partout) les travailleurs sont
lancés en mode "spawn" : votre script doit protéger son point d'entrée par
    if __name__ == "__main__":

Dépendances : numpy, scipy, rasterio
Hypothèses : MNT en CRS projeté (mètres), dépassant d'au moins ~2 km l'emprise d'étude.
"""

from __future__ import annotations

import itertools
import math
import multiprocessing
import os
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from scipy.signal import fftconvolve

NODATA = -9999.0
DEFAULT_LAPSE_RATE = -0.0065  # °C par mètre


# =============================================================================
# Infrastructure : profils, fenêtres, exécution parallèle
# =============================================================================
def _profile(height, width, count, transform, crs, nodata=NODATA):
    return dict(
        driver="GTiff", height=height, width=width, count=count,
        dtype="float32", crs=crs, transform=transform, nodata=nodata,
        compress="deflate", predictor=3, tiled=True,
        blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER",
    )


def _check_dem(src):
    if not src.crs or not src.crs.is_projected:
        raise ValueError("Le MNT doit être dans un CRS projeté (mètres). Utilisez reproject_dem().")
    if abs(src.crs.linear_units_factor[1] - 1.0) > 1e-9:
        raise ValueError("Les unités du CRS du MNT doivent être des mètres.")
    if not math.isclose(abs(src.transform.a), abs(src.transform.e), rel_tol=1e-3):
        raise ValueError("Pixels du MNT non carrés : rééchantillonnez-le.")


def _iter_windows(height, width, block):
    for r in range(0, height, block):
        for c in range(0, width, block):
            yield Window(c, r, min(block, width - c), min(block, height - r))


def _n_workers(n):
    return max(1, int(n)) if n else max(1, (os.cpu_count() or 2) - 1)


def _covering_window(bounds, transform, width, height, pad):
    """Fenêtre (entière, bornée au raster) d'une grille couvrant `bounds` + `pad` cellules."""
    w = from_bounds(*bounds, transform=transform)
    c0 = int(math.floor(w.col_off)) - pad
    r0 = int(math.floor(w.row_off)) - pad
    c1 = int(math.ceil(w.col_off + w.width)) + pad
    r1 = int(math.ceil(w.row_off + w.height)) + pad
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, width), min(r1, height)
    if c1 <= c0 or r1 <= r0:
        raise ValueError("Le MNT ne recouvre pas le raster CHELSA fourni.")
    return Window(c0, r0, c1 - c0, r1 - r0)


_CTX = None      # contexte partagé du travail en cours (défini dans chaque processus)
_COARSE = None   # cache mmap des données grossières


def _init_ctx(ctx):
    global _CTX, _COARSE
    _CTX, _COARSE = ctx, None


def _get_coarse():
    global _COARSE
    if _COARSE is None:
        _COARSE = np.load(_CTX["coarse_path"], mmap_mode="r")
    return _COARSE


def _run_parallel(func, windows, n_workers, ctx, on_result, desc=""):
    """
    Exécute func(window) -> (window, résultat) pour chaque fenêtre, en parallèle,
    avec au plus 2*n_workers tâches en vol (mémoire bornée). `on_result` est
    appelé dans le processus principal (c'est lui qui écrit).
    """
    windows = list(windows)
    total, done_n = len(windows), 0

    def tick():
        nonlocal done_n
        done_n += 1
        if total >= 10 and (done_n % max(1, total // 10) == 0 or done_n == total):
            print(f"{desc}: {done_n}/{total} blocs", flush=True)

    n = min(_n_workers(n_workers), total)
    if n <= 1:
        _init_ctx(ctx)
        for w in windows:
            on_result(*func(w))
            tick()
        return

    with ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_ctx, initargs=(ctx,)) as ex:
        it = iter(windows)
        pending = {ex.submit(func, w) for w in itertools.islice(it, 2 * n)}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                on_result(*f.result())
                tick()
                nxt = next(it, None)
                if nxt is not None:
                    pending.add(ex.submit(func, nxt))


def _to_nodata(a):
    return np.where(np.isfinite(a), a, NODATA).astype("float32")


# =============================================================================
# Préparation du MNT et lecture de CHELSA
# =============================================================================
def reproject_dem(src_path, dst_path, dst_crs="EPSG:2154", resolution=30.0,
                  resampling=Resampling.bilinear, n_threads=4):
    """
    Reprojette un MNT (ex. Copernicus GLO-30, WGS84) vers un CRS métrique à 30 m.
    GDAL écrit directement dans le fichier par morceaux : pas de chargement complet en mémoire.
    """
    with rasterio.open(src_path) as src:
        transform, w, h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution)
        with rasterio.open(dst_path, "w", **_profile(h, w, 1, transform, dst_crs)) as dst:
            reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                      src_nodata=src.nodata, dst_nodata=NODATA, resampling=resampling,
                      num_threads=n_threads, warp_mem_limit=512)
    return dst_path


def _read_window_covering(src_path, bounds, bounds_crs, scale=None, offset=None, pad=2):
    """Lit la fenêtre d'un raster (ex. CHELSA global) couvrant `bounds` (+ pad cellules)."""
    with rasterio.open(src_path) as src:
        b = transform_bounds(bounds_crs, src.crs, *bounds, densify_pts=21)
        win = _covering_window(b, src.transform, src.width, src.height, pad)
        a = src.read(1, window=win, masked=True).astype("float32")
        s = src.scales[0] if scale is None else scale
        o = src.offsets[0] if offset is None else offset
        return (a * s + o).filled(np.nan).astype("float32"), src.window_transform(win), src.crs


def resample_to_reference(src_path, ref_path, out_path, resampling=Resampling.bilinear,
                          scale=None, offset=None):
    """
    Rééchantillonne un raster (ex. précipitations CHELSA) sur la grille d'un raster de
    référence (le MNT), en écrivant par morceaux. Utile uniquement avec compute_bioclim() ;
    downscale_bioclim() n'en a pas besoin.
    """
    with rasterio.open(ref_path) as ref:
        h, w, tr, crs, bounds = ref.height, ref.width, ref.transform, ref.crs, ref.bounds
    a, s_tr, s_crs = _read_window_covering(src_path, bounds, crs, scale, offset)
    with rasterio.open(out_path, "w", **_profile(h, w, 1, tr, crs)) as dst:
        reproject(a, rasterio.band(dst, 1), src_transform=s_tr, src_crs=s_crs,
                  src_nodata=np.nan, dst_nodata=NODATA, resampling=resampling)
    return out_path


# =============================================================================
# 1. Downscaling de la température (par blocs, parallèle)
# =============================================================================
def estimate_lapse_rate(t_coarse, dem_coarse, bounds=(-0.0090, -0.0030)):
    """
    Gradient altitudinal régional (°C/m) par régression T ~ altitude sur les cellules
    CHELSA de la zone, borné par sécurité (inversions hivernales). Retombe sur
    -0.0065 si le relief est insuffisant (< 100 m de dénivelé).
    """
    ok = np.isfinite(t_coarse) & np.isfinite(dem_coarse)
    if ok.sum() < 30 or np.ptp(dem_coarse[ok]) < 100:
        return DEFAULT_LAPSE_RATE
    return float(np.clip(np.polyfit(dem_coarse[ok], t_coarse[ok], 1)[0], *bounds))


def _prepare_coarse(paths, units, dem_path, pad_cells, n_threads):
    """
    Données grossières, petites, calculées une seule fois :
      - stack (N, h, w) : couches CHELSA converties (°C ou mm), sur la fenêtre CHELSA du MNT ;
      - z_c (h, w)      : altitude moyenne du MNT par cellule CHELSA (moyenne calculée en
                          streaming par GDAL à partir du fichier MNT).
    """
    with rasterio.open(dem_path) as dem:
        _check_dem(dem)
        layers, ref = [], None
        for p, (s, o) in zip(paths, units):
            a, tr, crs = _read_window_covering(p, dem.bounds, dem.crs, s, o, pad_cells)
            if ref is None:
                ref = (a.shape, tr, crs)
            elif a.shape != ref[0] or not tr.almost_equals(ref[1]):
                raise ValueError(f"{p} n'est pas sur la même grille que {paths[0]}.")
            layers.append(a)
        stack = np.stack(layers)
        z_c = np.full(ref[0], np.nan, dtype="float32")
        reproject(rasterio.band(dem, 1), z_c, dst_transform=ref[1], dst_crs=ref[2],
                  resampling=Resampling.average, src_nodata=dem.nodata, dst_nodata=np.nan,
                  num_threads=n_threads, warp_mem_limit=512)
    z_c[~np.isfinite(stack[0])] = np.nan
    return stack, z_c, ref[1], ref[2]


def _resolve_gammas(gamma, stack, z_c, is_temp):
    n = len(stack)
    if isinstance(gamma, str):
        if gamma != "auto":
            raise ValueError('gamma doit être un nombre, une liste ou "auto".')
        g = [estimate_lapse_rate(stack[i], z_c) if is_temp[i] else 0.0 for i in range(n)]
    elif np.ndim(gamma) == 0:
        g = [float(gamma)] * n
    else:
        g = list(gamma)
        if len(g) != n:
            raise ValueError("Longueur de `gamma` incompatible avec le nombre de rasters.")
    g = np.array(g, dtype="float32")
    g[~np.asarray(is_temp)] = 0.0  # pas de correction altitudinale pour les précipitations
    return g


def _fine_block(win):
    """
    Travailleur : pour un bloc du MNT, interpole (bilinéaire) les couches CHELSA et
    l'altitude moyenne de cellule, puis applique T += gamma * (z - z_ref).
    Retourne (fenêtre, tableau (N, h, w)).
    """
    c = _CTX
    coarse = _get_coarse()  # (N+1, h, w) ; dernière couche = altitude moyenne
    with rasterio.open(c["dem_path"]) as src:
        dem = src.read(1, window=win, masked=True).astype("float32").filled(np.nan)
        tr, crs, b = src.window_transform(win), src.crs, src.window_bounds(win)

    cb = transform_bounds(crs, c["c_crs"], *b, densify_pts=21)
    sub = _covering_window(cb, c["c_tr"], coarse.shape[2], coarse.shape[1], pad=2)
    r0, c0 = int(sub.row_off), int(sub.col_off)
    part = np.asarray(coarse[:, r0:r0 + int(sub.height), c0:c0 + int(sub.width)])

    dst = np.full((part.shape[0], int(win.height), int(win.width)), np.nan, dtype="float32")
    reproject(part, dst, src_transform=window_transform(sub, c["c_tr"]), src_crs=c["c_crs"],
              dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    fine, z_ref = dst[:-1], dst[-1]
    fine += c["gammas"][:, None, None] * (dem - z_ref)[None]
    return win, fine


def downscale_temperature_many(chelsa_paths: Sequence[str], dem_path: str,
                               out_paths: Sequence[str],
                               gamma: float | str | Sequence[float] = DEFAULT_LAPSE_RATE,
                               scale=0.1, offset=-273.15, block_size=512,
                               n_workers=None, pad_cells=2):
    """
    Downscale plusieurs rasters de température CHELSA (même grille) sur la grille du MNT :

        T_fine = T_chelsa(bilinéaire) + gamma * (z_30m - z_moyen_cellule(bilinéaire))

    Le MNT et l'altitude de référence ne sont traités qu'une fois pour tous les rasters.

    gamma : °C/m (scalaire, liste de même longueur que chelsa_paths, ou "auto" =
            régression régionale par raster).
    scale/offset : CHELSA v2.1 stocke des Kelvin x 10 -> 0.1 et -273.15 (vérifier avec
            gdalinfo ; None = lire dans les métadonnées du fichier).
    block_size : multiple de 256 recommandé. n_workers : défaut = nb de CPU - 1.
    Retourne (out_paths, gammas utilisés).
    """
    if len(chelsa_paths) != len(out_paths):
        raise ValueError("chelsa_paths et out_paths doivent avoir la même longueur.")
    n_threads = _n_workers(n_workers)
    with tempfile.TemporaryDirectory() as tmp:
        stack, z_c, c_tr, c_crs = _prepare_coarse(
            chelsa_paths, [(scale, offset)] * len(chelsa_paths), dem_path, pad_cells, n_threads)
        gammas = _resolve_gammas(gamma, stack, z_c, np.ones(len(stack), bool))
        path = os.path.join(tmp, "coarse.npy")
        np.save(path, np.concatenate([stack, z_c[None]]).astype("float32"))
        ctx = dict(dem_path=dem_path, coarse_path=path, c_tr=c_tr, c_crs=c_crs, gammas=gammas)

        with rasterio.open(dem_path) as d:
            H, W, tr, crs = d.height, d.width, d.transform, d.crs
        dsts = [rasterio.open(p, "w", **_profile(H, W, 1, tr, crs)) for p in out_paths]
        try:
            def on_result(win, fine):
                out = _to_nodata(fine)
                for i, ds in enumerate(dsts):
                    ds.write(out[i], 1, window=win)
            _run_parallel(_fine_block, _iter_windows(H, W, block_size), n_workers, ctx,
                          on_result, "downscale")
        finally:
            for ds in dsts:
                ds.close()
    return list(out_paths), gammas


def downscale_temperature(chelsa_path, dem_path, out_path, gamma=DEFAULT_LAPSE_RATE,
                          scale=0.1, offset=-273.15, block_size=512, n_workers=None, pad_cells=2):
    """Version un-seul-raster de downscale_temperature_many(). Retourne (out_path, gamma)."""
    _, g = downscale_temperature_many([chelsa_path], dem_path, [out_path], gamma, scale, offset,
                                      block_size, n_workers, pad_cells)
    return out_path, float(g[0])


# =============================================================================
# 2. Variables bioclimatiques (températures)
# =============================================================================
def _quarter_mean(a):
    """a : (12, H, W) -> moyenne glissante de 3 mois (circulaire), trimestre débutant au mois i."""
    return (a + np.roll(a, -1, axis=0) + np.roll(a, -2, axis=0)) / 3.0


def bioclim_from_arrays(tas, tasmin, tasmax, pr=None):
    """
    Calcule bio1-bio7, bio10, bio11 (et bio8, bio9 si `pr` est fourni).
    Entrées : tableaux (12, H, W), °C (pr en mm), mois 1 -> 12.
      bio1  T moyenne annuelle              bio2  amplitude diurne moyenne
      bio3  isothermalité (bio2/bio7*100)   bio4  saisonnalité (écart-type tas * 100)
      bio5  tmax du mois le plus chaud      bio6  tmin du mois le plus froid
      bio7  amplitude annuelle (bio5-bio6)  bio8/bio9 T du trimestre le plus humide/sec
      bio10 T du trimestre le plus chaud    bio11 T du trimestre le plus froid
    """
    valid = np.isfinite(tas).all(0) & np.isfinite(tasmin).all(0) & np.isfinite(tasmax).all(0)
    if pr is not None:
        valid &= np.isfinite(pr).all(0)

    bio = {}
    bio["bio1"] = tas.mean(0)
    bio["bio2"] = (tasmax - tasmin).mean(0)
    bio["bio5"] = tasmax.max(0)
    bio["bio6"] = tasmin.min(0)
    bio["bio7"] = bio["bio5"] - bio["bio6"]
    with np.errstate(divide="ignore", invalid="ignore"):
        bio["bio3"] = np.where(bio["bio7"] != 0, bio["bio2"] / bio["bio7"] * 100.0, np.nan)
    bio["bio4"] = tas.std(0) * 100.0

    tq = _quarter_mean(np.where(valid, tas, 0.0))
    pick = lambda idx: np.take_along_axis(tq, idx[None], axis=0)[0]
    bio["bio10"] = pick(tq.argmax(0))
    bio["bio11"] = pick(tq.argmin(0))
    if pr is not None:
        pq = _quarter_mean(np.where(valid, pr, 0.0))
        bio["bio8"] = pick(pq.argmax(0))
        bio["bio9"] = pick(pq.argmin(0))
    return {k: np.where(valid, v, np.nan).astype("float32") for k, v in bio.items()}


def _bioclim_names(has_pr):
    return (["bio1", "bio2", "bio3", "bio4", "bio5", "bio6", "bio7"]
            + (["bio8", "bio9"] if has_pr else []) + ["bio10", "bio11"])


def _bioclim_downscaled_block(win):
    """Travailleur de downscale_bioclim() : downscaling + bioclim en mémoire, sans fichier intermédiaire."""
    win, fine = _fine_block(win)
    pr = fine[36:48] if _CTX["has_pr"] else None
    bio = bioclim_from_arrays(fine[0:12], fine[12:24], fine[24:36], pr)
    return win, np.stack([bio[n] for n in _CTX["names"]])


def downscale_bioclim(tas_files: Sequence[str], tasmin_files: Sequence[str],
                      tasmax_files: Sequence[str], dem_path: str, out_path: str,
                      pr_files: Sequence[str] | None = None,
                      gamma: float | str = "auto", scale=0.1, offset=-273.15,
                      pr_scale=None, pr_offset=None,
                      block_size=512, n_workers=None, pad_cells=2):
    """
    Chaîne complète en une passe, par blocs et en parallèle :
    CHELSA mensuel BRUT (12 tas, 12 tasmin, 12 tasmax, [12 pr]) + MNT 30 m
    -> GeoTIFF multibande bio1..bio11 à 30 m. Aucun raster mensuel intermédiaire
    n'est écrit (économie de dizaines de Go sur une grande emprise).

    Les listes sont à fournir dans l'ordre janvier -> décembre. Les précipitations
    sont simplement interpolées (aucune correction altitudinale) ; seul leur classement
    entre mois compte pour bio8/bio9. pr_scale/pr_offset=None -> métadonnées du fichier.
    """
    lists = [tas_files, tasmin_files, tasmax_files] + ([pr_files] if pr_files else [])
    if any(len(l) != 12 for l in lists):
        raise ValueError("12 fichiers mensuels attendus par variable.")
    paths = [p for l in lists for p in l]
    units = [(scale, offset)] * 36 + ([(pr_scale, pr_offset)] * 12 if pr_files else [])
    is_temp = np.array([True] * 36 + ([False] * 12 if pr_files else []))
    names = _bioclim_names(bool(pr_files))
    n_threads = _n_workers(n_workers)

    with tempfile.TemporaryDirectory() as tmp:
        stack, z_c, c_tr, c_crs = _prepare_coarse(paths, units, dem_path, pad_cells, n_threads)
        gammas = _resolve_gammas(gamma, stack, z_c, is_temp)
        path = os.path.join(tmp, "coarse.npy")
        np.save(path, np.concatenate([stack, z_c[None]]).astype("float32"))
        ctx = dict(dem_path=dem_path, coarse_path=path, c_tr=c_tr, c_crs=c_crs, gammas=gammas,
                   has_pr=bool(pr_files), names=names)

        with rasterio.open(dem_path) as d:
            H, W, tr, crs = d.height, d.width, d.transform, d.crs
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **_profile(H, W, len(names), tr, crs)) as dst:
            for i, n in enumerate(names, start=1):
                dst.set_band_description(i, n)
            _run_parallel(_bioclim_downscaled_block, _iter_windows(H, W, block_size), n_workers,
                          ctx, lambda win, a: dst.write(_to_nodata(a), window=win), "bioclim")
    return out_path, names


def _bioclim_files_block(win):
    """Travailleur de compute_bioclim() : lit la fenêtre dans les rasters mensuels."""
    def read(files):
        out = []
        for f in files:
            with rasterio.open(f) as s:
                out.append(s.read(1, window=win, masked=True).astype("float32").filled(np.nan))
        return np.stack(out)
    c = _CTX
    pr = read(c["pr"]) if c["pr"] else None
    bio = bioclim_from_arrays(read(c["tas"]), read(c["tasmin"]), read(c["tasmax"]), pr)
    return win, np.stack([bio[n] for n in c["names"]])


def compute_bioclim(tas_files: Sequence[str], tasmin_files: Sequence[str],
                    tasmax_files: Sequence[str], out_path: str,
                    pr_files: Sequence[str] | None = None, block_size=512, n_workers=None):
    """
    Bioclim à partir de 12 rasters mensuels DÉJÀ sur la même grille (ex. sorties de
    downscale_temperature_many ; pr via resample_to_reference). Par blocs, en parallèle.
    Pour partir de CHELSA brut, préférez downscale_bioclim().
    """
    lists = [tas_files, tasmin_files, tasmax_files] + ([pr_files] if pr_files else [])
    if any(len(l) != 12 for l in lists):
        raise ValueError("12 fichiers mensuels attendus par variable.")
    names = _bioclim_names(bool(pr_files))
    with rasterio.open(tas_files[0]) as ref:
        H, W, tr, crs = ref.height, ref.width, ref.transform, ref.crs
    ctx = dict(tas=list(tas_files), tasmin=list(tasmin_files), tasmax=list(tasmax_files),
               pr=list(pr_files) if pr_files else None, names=names)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **_profile(H, W, len(names), tr, crs)) as dst:
        for i, n in enumerate(names, start=1):
            dst.set_band_description(i, n)
        _run_parallel(_bioclim_files_block, _iter_windows(H, W, block_size), n_workers, ctx,
                      lambda win, a: dst.write(_to_nodata(a), window=win), "bioclim")
    return out_path, names


# =============================================================================
# 3 & 4. Terrain : pente, orientation, TPI (par blocs avec halo)
# =============================================================================
def slope_aspect(dem, xres, yres):
    """
    Pente et orientation par la méthode de Horn (identique à gdaldem).
    Retourne (pente en degrés, orientation 0-360° horaire depuis le nord ; NaN si plat).
    """
    p = np.pad(dem, 1, mode="edge")
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * abs(xres))   # vers l'est
    dzdn = ((a + 2 * b + c) - (g + 2 * h + i)) / (8.0 * abs(yres))   # vers le nord
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdn)))
    aspect = np.degrees(np.arctan2(-dzdx, -dzdn)) % 360.0
    aspect = np.where(np.hypot(dzdx, dzdn) == 0, np.nan, aspect)
    return slope.astype("float32"), aspect.astype("float32")


def tpi(dem, radius_m, res):
    """
    TPI = altitude - altitude moyenne du voisinage circulaire de rayon `radius_m`
    (pixel central exclu). Gère NaN et bords. Convolution par FFT.
    """
    valid = np.isfinite(dem)
    if not valid.any():
        return np.full(dem.shape, np.nan, dtype="float32")
    r = radius_m / res
    ri = int(round(r))
    y, x = np.ogrid[-ri:ri + 1, -ri:ri + 1]
    kernel = ((x * x + y * y) <= r * r).astype("float32")
    kernel[ri, ri] = 0.0
    z = np.where(valid, dem - np.nanmean(dem), 0.0).astype("float32")
    num = fftconvolve(z, kernel, mode="same")
    den = fftconvolve(valid.astype("float32"), kernel, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_z = np.where(den > 0.5, num / den, np.nan)
    out = (z - mean_z).astype("float32")
    out[~valid] = np.nan
    return out


def _terrain_bands(dem, res, radii):
    slope, aspect = slope_aspect(dem, res, res)
    flat, valid = ~np.isfinite(aspect), np.isfinite(dem)
    bands = {
        "elevation": dem, "slope": slope, "aspect": aspect,
        "northness": np.where(valid, np.where(flat, 0.0, np.cos(np.radians(aspect))), np.nan),
        "eastness": np.where(valid, np.where(flat, 0.0, np.sin(np.radians(aspect))), np.nan),
    }
    for r in radii:
        bands[f"tpi_{int(r)}m"] = tpi(dem, r, res)
    return bands


def _terrain_block(win):
    """Travailleur : lit le bloc + halo, calcule, rogne le halo."""
    c, halo = _CTX, _CTX["halo"]
    with rasterio.open(c["dem_path"]) as src:
        H, W = src.height, src.width
        r0, c0 = max(int(win.row_off) - halo, 0), max(int(win.col_off) - halo, 0)
        r1 = min(int(win.row_off + win.height) + halo, H)
        c1 = min(int(win.col_off + win.width) + halo, W)
        dem = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), masked=True)
        dem = dem.astype("float32").filled(np.nan)
    bands = _terrain_bands(dem, c["res"], c["radii"])
    dr, dc = int(win.row_off) - r0, int(win.col_off) - c0
    h, w = int(win.height), int(win.width)
    return win, np.stack([np.asarray(b[dr:dr + h, dc:dc + w], dtype="float32")
                          for b in bands.values()])


def compute_terrain(dem_path, out_path, tpi_radii_m: Sequence[float] = (300, 750, 1500),
                    block_size=1024, n_workers=None):
    """
    GeoTIFF multibande sur la grille du MNT : elevation, slope (°), aspect (°, NaN si plat),
    northness (cos), eastness (sin), tpi_<rayon>m. Par blocs avec halo, en parallèle ;
    le résultat est identique à un calcul en un seul bloc.

    northness/eastness sont préférables à l'orientation brute pour un réseau de neurones
    (l'angle est circulaire) ; ils valent 0 sur terrain plat.
    """
    with rasterio.open(dem_path) as src:
        _check_dem(src)
        H, W, tr, crs, res = src.height, src.width, src.transform, src.crs, abs(src.transform.a)
    halo = int(math.ceil(max(tpi_radii_m) / res)) + 2
    names = ["elevation", "slope", "aspect", "northness", "eastness"] + \
            [f"tpi_{int(r)}m" for r in tpi_radii_m]
    ctx = dict(dem_path=dem_path, halo=halo, res=res, radii=tuple(tpi_radii_m))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **_profile(H, W, len(names), tr, crs)) as dst:
        for i, n in enumerate(names, start=1):
            dst.set_band_description(i, n)
        _run_parallel(_terrain_block, _iter_windows(H, W, block_size), n_workers, ctx,
                      lambda win, a: dst.write(_to_nodata(a), window=win), "terrain")
    return out_path, names


# =============================================================================
# Exemple de chaîne complète
# =============================================================================
if __name__ == "__main__":   # garde OBLIGATOIRE (multiprocessing en mode spawn)
    # --- À ADAPTER -----------------------------------------------------------
    CHELSA = Path("chelsa")
    DEM_RAW, DEM = "dem_glo30_wgs84.tif", "dem_30m_epsg2154.tif"
    OUT = Path("out")
    N_WORKERS = 6
    # -------------------------------------------------------------------------
    f = lambda var: [str(CHELSA / f"CHELSA_{var}_{m:02d}_1981-2010_V.2.1.tif") for m in range(1, 13)]

    reproject_dem(DEM_RAW, DEM, dst_crs="EPSG:2154", resolution=30)

    # CHELSA brut -> bio1..bio11 à 30 m, en une passe (sans fichiers mensuels intermédiaires)
    downscale_bioclim(f("tas"), f("tasmin"), f("tasmax"), DEM, OUT / "bioclim_30m.tif",
                      pr_files=f("pr"), gamma="auto", n_workers=N_WORKERS)

    compute_terrain(DEM, OUT / "terrain_30m.tif", tpi_radii_m=(300, 750, 1500),
                    n_workers=N_WORKERS)

    # Si vous voulez AUSSI les températures mensuelles à 30 m :
    # downscale_temperature_many(f("tas"), DEM, [OUT / f"tas_{m:02d}_30m.tif" for m in range(1, 13)],
    #                            gamma="auto", n_workers=N_WORKERS)
