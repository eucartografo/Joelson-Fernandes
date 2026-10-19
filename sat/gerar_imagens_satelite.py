# -*- coding: utf-8 -*-
"""
Gera recortes de satelite (Sentinel-2, publico/gratuito) para os imoveis monitorados.
Roda mensalmente (idempotente): busca a cena mais recente com pouca nuvem (mosaicando
tiles adjacentes quando o imovel cruza a borda de duas cenas) e sobrescreve os PNGs
+ status.json de cada imovel.
"""
import json
import os
import datetime
import numpy as np
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling, calculate_default_transform
from rasterio.merge import merge as rio_merge
from rasterio.io import MemoryFile
from PIL import Image
import urllib.request

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
OUT_DIR = "sat"
BUFFER_DEG = 0.01  # ~1.1 km de margem ao redor do imovel

PROPRIEDADES = {
    "8749": {
        "nome": "Fazenda Piquizeiro - Gleba I",
        "bbox": [-45.69070556, -14.49362639, -45.5997675, -14.38513306],
    },
    "8750": {
        "nome": "Fazenda Piquizeiro - Gleba II",
        "bbox": [-45.71641944, -14.51821667, -45.65433306, -14.4231725],
    },
}


def buffered_bbox(bbox, buf=BUFFER_DEG):
    minx, miny, maxx, maxy = bbox
    return [minx - buf, miny - buf, maxx + buf, maxy + buf]


def stac_search(bbox, max_cloud, days_back):
    end = datetime.datetime.now(datetime.timezone.utc)
    start = end - datetime.timedelta(days=days_back)
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{start.strftime('%Y-%m-%dT00:00:00Z')}/{end.strftime('%Y-%m-%dT23:59:59Z')}",
        "limit": 100,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    req = urllib.request.Request(
        STAC_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data.get("features", [])


def best_date_group(features, recent_window_days=30):
    """Agrupa por data (dia). Entre os dias dentro da janela recente, escolhe o de
    menor nuvem media (evita pegar cena com nuvem quando existe uma mais limpa a
    poucos dias de distancia). Se nenhum dia recente tiver nuvem baixa, cai para o
    de menor nuvem em toda a busca."""
    by_date = {}
    for f in features:
        day = f["properties"]["datetime"][:10]
        by_date.setdefault(day, []).append(f)

    today = datetime.datetime.now(datetime.timezone.utc).date()
    scored = []
    for day, group in by_date.items():
        avg_cloud = sum(f["properties"]["eo:cloud_cover"] for f in group) / len(group)
        age_days = (today - datetime.date.fromisoformat(day)).days
        scored.append((day, group, avg_cloud, age_days))

    recent = [s for s in scored if s[3] <= recent_window_days]
    pool = recent if recent else scored
    day, group, avg_cloud, _ = min(pool, key=lambda s: s[2])
    return day, group, avg_cloud


def merged_crop_wgs84(hrefs, bbox_wgs84, dst_crs="EPSG:4326"):
    srcs = [rasterio.open(h) for h in hrefs]
    try:
        src_crs = srcs[0].crs
        left, bottom, right, top = transform_bounds("EPSG:4326", src_crs, *bbox_wgs84)
        merged, merged_transform = rio_merge(srcs, bounds=(left, bottom, right, top))
    finally:
        for s in srcs:
            s.close()

    bands, height, width = merged.shape
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height,
        left=left, bottom=bottom, right=right, top=top,
    )
    dst = np.zeros((bands, dst_height, dst_width), dtype=merged.dtype)
    for i in range(bands):
        reproject(
            source=merged[i], destination=dst[i],
            src_transform=merged_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    bounds_wgs84 = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)
    return dst, bounds_wgs84


def save_tci_png(rgb_array, path):
    arr = np.transpose(rgb_array, (1, 2, 0)).astype("float32")
    valid = arr[arr > 0]
    if valid.size:
        lo, hi = np.percentile(valid, [1, 99])
        if hi <= lo:
            hi = lo + 1
    else:
        lo, hi = 0, 255
    stretched = np.clip((arr - lo) / (hi - lo), 0, 1)
    stretched = np.power(stretched, 0.9)
    out = (stretched * 255).astype("uint8")
    Image.fromarray(out, mode="RGB").save(path)


def save_ndvi_png(red, nir, path):
    red = red.astype("float32")
    nir = nir.astype("float32")
    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / (denom + 1e-6))
    valid_mask = ~np.isnan(ndvi) & (denom > 0)
    import matplotlib.cm as cm
    norm = np.clip((np.nan_to_num(ndvi) + 1) / 2, 0, 1)
    colored = (cm.RdYlGn(norm)[:, :, :3] * 255).astype("uint8")
    colored[~valid_mask] = 255
    Image.fromarray(colored, mode="RGB").save(path)
    vals = ndvi[valid_mask]
    if vals.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    status = {}

    for mat, info in PROPRIEDADES.items():
        print(f"=== Matricula {mat} ===")
        bbox = buffered_bbox(info["bbox"])
        feats = stac_search(bbox, max_cloud=40, days_back=90)
        if not feats:
            print("  nenhuma cena encontrada!")
            status[mat] = {"erro": "nenhuma cena Sentinel-2 encontrada nos ultimos 90 dias"}
            continue

        day, group, avg_cloud = best_date_group(feats)
        print(f"  data: {day} | tiles: {[f['id'] for f in group]} | nuvem media: {round(avg_cloud,2)}%")

        tci_hrefs = [f["assets"]["visual"]["href"] for f in group]
        red_hrefs = [f["assets"]["red"]["href"] for f in group]
        nir_hrefs = [f["assets"]["nir"]["href"] for f in group]

        tci_data, bounds = merged_crop_wgs84(tci_hrefs, bbox)
        save_tci_png(tci_data, f"{OUT_DIR}/mat{mat}_tci.png")

        red_data, _ = merged_crop_wgs84(red_hrefs, bbox)
        nir_data, _ = merged_crop_wgs84(nir_hrefs, bbox)
        ndvi_mean, ndvi_min, ndvi_max = save_ndvi_png(red_data[0], nir_data[0], f"{OUT_DIR}/mat{mat}_ndvi.png")

        status[mat] = {
            "nome": info["nome"],
            "cenas": [f["id"] for f in group],
            "data_imagem": day,
            "nuvem_pct": round(avg_cloud, 2),
            "ndvi_medio": round(ndvi_mean, 4),
            "ndvi_min": round(ndvi_min, 4),
            "ndvi_max": round(ndvi_max, 4),
            "bounds_wgs84": list(bounds),
            "gerado_em": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        print("  bounds:", [round(x, 4) for x in bounds], "| NDVI medio:", round(ndvi_mean, 4))

    with open(f"{OUT_DIR}/status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    print("Status salvo em", f"{OUT_DIR}/status.json")


if __name__ == "__main__":
    main()
