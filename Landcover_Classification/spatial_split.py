from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def clean_labels(gdf: gpd.GeoDataFrame, label_col: str) -> gpd.GeoDataFrame:
    out = gdf.dropna(subset=[label_col, 'geometry']).copy()
    out = out[~out.geometry.is_empty].copy()
    out = out[~out[label_col].isin([-1, -1.0, ''])].copy()
    if out.crs is None or not out.crs.is_projected:
        raise ValueError('Use a projected CRS with metre units, e.g. UTM.')
    out = out.reset_index(drop=False).rename(columns={'index': 'src_id'})
    out['poly_id'] = np.arange(len(out), dtype=np.int64)
    return out


def add_spatial_blocks(gdf: gpd.GeoDataFrame, block_size_m: float) -> gpd.GeoDataFrame:
    out = gdf.copy()
    centroids = out.geometry.centroid
    min_x, min_y, _, _ = out.total_bounds
    out['blk_x'] = np.floor((centroids.x - min_x) / block_size_m).astype(int)
    out['blk_y'] = np.floor((centroids.y - min_y) / block_size_m).astype(int)
    out['block_id'] = out['blk_x'].astype(str) + '_' + out['blk_y'].astype(str)
    return out


def split_score(full, held, label_col, target_fraction):
    full_classes = set(full[label_col].unique())
    held_classes = set(held[label_col].unique())
    missing = len(full_classes - held_classes)
    frac_error = abs(len(held) / len(full) - target_fraction)
    p_full = full[label_col].value_counts(normalize=True).sort_index()
    p_held = held[label_col].value_counts(normalize=True).reindex(p_full.index, fill_value=0.0)
    class_drift = float(np.abs(p_full - p_held).mean())
    return missing * 100.0 + frac_error * 10.0 + class_drift


def choose_group_holdout(gdf, label_col, fraction, seed, n_candidates=250):
    best = None
    groups = gdf['block_id'].to_numpy()
    y = gdf[label_col].to_numpy()
    for i in range(n_candidates):
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + i)
        keep_idx, held_idx = next(splitter.split(gdf, y=y, groups=groups))
        keep, held = gdf.iloc[keep_idx].copy(), gdf.iloc[held_idx].copy()
        score = split_score(gdf, held, label_col, fraction)
        if best is None or score < best[0]:
            best = (score, keep, held)
    return best[1], best[2]


def remove_near_heldout(candidate_train, heldout, buffer_m):
    if buffer_m <= 0 or heldout.empty:
        return candidate_train.copy()
    held_buffer = heldout[['geometry']].copy()
    held_buffer['geometry'] = held_buffer.geometry.buffer(buffer_m)
    held_union = held_buffer.geometry.union_all()
    return candidate_train.loc[~candidate_train.geometry.intersects(held_union)].copy()


def assert_class_coverage(splits, label_col):
    all_classes = set(pd.concat([g[label_col] for g in splits.values()]).unique())
    problems = {name: sorted(all_classes - set(g[label_col].unique())) for name, g in splits.items()}
    problems = {k: v for k, v in problems.items() if v}
    if problems:
        raise ValueError(f'Splits missing classes: {problems}. Try smaller blocks/buffer or another seed.')


def create_spatial_train_val_test(labels_path, output_dir, label_col='Final_Clas', block_size_m=25.0,
                                  test_fraction=0.20, val_fraction_of_remaining=0.20,
                                  guard_buffer_m=5.0, seed=42):
    labels_path, output_dir = Path(labels_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gdf = add_spatial_blocks(clean_labels(gpd.read_file(labels_path), label_col), block_size_m)

    development, test = choose_group_holdout(gdf, label_col, test_fraction, seed)
    development = remove_near_heldout(development, test, guard_buffer_m)
    train, validation = choose_group_holdout(development, label_col, val_fraction_of_remaining, seed + 10000)
    train = remove_near_heldout(train, validation, guard_buffer_m)
    validation = remove_near_heldout(validation, test, guard_buffer_m)

    splits = {'train': train, 'validation': validation, 'test': test}
    assert_class_coverage(splits, label_col)

    summary = {}
    for name, split in splits.items():
        path = output_dir / f'{name}.gpkg'
        split.to_file(path, layer=name, driver='GPKG')
        summary[name] = {
            'n_polygons': int(len(split)),
            'n_blocks': int(split['block_id'].nunique()),
            'class_counts': {str(k): int(v) for k, v in split[label_col].value_counts().sort_index().items()},
        }
    manifest = {
        'source': str(labels_path), 'label_col': label_col, 'block_size_m': block_size_m,
        'test_fraction': test_fraction, 'val_fraction_of_remaining': val_fraction_of_remaining,
        'guard_buffer_m': guard_buffer_m, 'seed': seed, 'summary': summary,
    }
    (output_dir / 'split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    create_spatial_train_val_test(
        labels_path='../Data/9_Training_Data/training_data_final.shp',
        output_dir='../Data/9_Training_Data/spatial_split_v4',
        block_size_m=25.0,
        guard_buffer_m=5.0,
    )
