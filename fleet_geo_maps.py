#!/usr/bin/env python3
"""
fleet_geo_maps.py — Geographic visualizations of fleet detection results.

Produces:
  output/fleet_map_static.png   — globe map of fleet event centroids (white scheme)
  output/fleet_map_convoys.png  — convoy start locations coloured by duration
  output/fleet_map_interactive.html — interactive Folium map
"""
import json
import os
import sqlite3

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import natural_earth, Reader
import folium
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

DB_PATH    = "whaling.db"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tables():
    con = sqlite3.connect(DB_PATH)
    fleet  = pd.read_sql("SELECT * FROM fleet_events",  con)
    convoy = pd.read_sql("SELECT * FROM fleet_convoys", con)
    con.close()
    return fleet, convoy


def build_land_filter():
    """Return a prepared shapely geometry for fast ocean/land testing."""
    shp = natural_earth(resolution='110m', category='physical', name='land')
    geom = unary_union(list(Reader(shp).geometries()))
    return prep(geom)


def filter_ocean(df, land, lat_col='centroid_lat', lon_col='centroid_lon'):
    """Drop rows whose centroid falls on land."""
    mask = [not land.contains(Point(lon, lat))
            for lon, lat in zip(df[lon_col], df[lat_col])]
    return df[mask].copy()


# ── 1. Static fleet-event globe map (white scheme) ───────────────────────────

def plot_fleet_map(fleet, land):
    proj = ccrs.Robinson()
    fig  = plt.figure(figsize=(18, 10), facecolor='white')
    ax   = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_facecolor('#dceeff')          # pale ocean blue

    ax.add_feature(cfeature.LAND,       facecolor='#f2f0eb', zorder=1)
    ax.add_feature(cfeature.OCEAN,      facecolor='#dceeff', zorder=0)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.5, edgecolor='#999999', zorder=2)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.3, edgecolor='#bbbbbb', zorder=2)
    ax.add_feature(cfeature.LAKES,      facecolor='#dceeff', zorder=2)
    ax.gridlines(color='#cccccc', linewidth=0.3, linestyle='--', zorder=1)
    ax.set_global()

    ocean_fleet = filter_ocean(fleet, land)
    removed = len(fleet) - len(ocean_fleet)
    print(f"  Removed {removed} mainland centroids → {len(ocean_fleet):,} ocean events")

    vmin, vmax = 2, ocean_fleet['n_vessels'].max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps['plasma']
    sizes = np.clip(ocean_fleet['n_obs'] * 1.5, 8, 120)

    sc = ax.scatter(
        ocean_fleet['centroid_lon'], ocean_fleet['centroid_lat'],
        c=ocean_fleet['n_vessels'], cmap=cmap, norm=norm,
        s=sizes, alpha=0.75, linewidths=0.25, edgecolors='#444444',
        transform=ccrs.PlateCarree(), zorder=4
    )

    top10 = ocean_fleet.nlargest(10, 'n_vessels')
    ax.scatter(
        top10['centroid_lon'], top10['centroid_lat'],
        s=sizes[top10.index] + 80, facecolors='none',
        edgecolors='#cc0000', linewidths=1.2,
        transform=ccrs.PlateCarree(), zorder=5
    )
    for _, row in top10.iterrows():
        ax.annotate(
            f"{int(row['n_vessels'])}v",
            xy=proj.transform_point(row['centroid_lon'], row['centroid_lat'],
                                    ccrs.PlateCarree()),
            xytext=(5, 5), textcoords='offset points',
            color='#cc0000', fontsize=7, fontweight='bold',
            zorder=6
        )

    val = ocean_fleet[ocean_fleet['validated_by_spoke'] == 1]
    if len(val):
        ax.scatter(
            val['centroid_lon'], val['centroid_lat'],
            marker='*', s=45, color='#007700', alpha=0.9,
            label=f'Spoke-validated ({len(val)})',
            transform=ccrs.PlateCarree(), zorder=6
        )
        ax.legend(loc='lower left', framealpha=0.8, fontsize=9,
                  facecolor='white', edgecolor='#cccccc')

    cb = fig.colorbar(sc, ax=ax, orientation='vertical',
                      pad=0.02, shrink=0.65, aspect=25)
    cb.set_label('Vessels co-located', fontsize=10, color='#333333')
    cb.ax.tick_params(labelsize=8, color='#333333')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#333333')

    ax.set_title(
        f'Fleet-in-Company Events  ·  {len(ocean_fleet):,} ocean clusters  ·  '
        f'up to {int(ocean_fleet["n_vessels"].max())} vessels co-located',
        fontsize=13, color='#222222', pad=14
    )

    out = os.path.join(OUTPUT_DIR, 'fleet_map_static.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


# ── 2. Convoy start-location globe map (white scheme) ────────────────────────

def plot_convoy_map(convoy, land):
    proj = ccrs.Robinson()
    fig  = plt.figure(figsize=(18, 10), facecolor='white')
    ax   = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_facecolor('#dceeff')

    ax.add_feature(cfeature.LAND,      facecolor='#f2f0eb', zorder=1)
    ax.add_feature(cfeature.OCEAN,     facecolor='#dceeff', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#999999', zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor='#bbbbbb', zorder=2)
    ax.add_feature(cfeature.LAKES,     facecolor='#dceeff', zorder=2)
    ax.gridlines(color='#cccccc', linewidth=0.3, linestyle='--', zorder=1)
    ax.set_global()

    ocean_convoy = filter_ocean(convoy, land,
                                lat_col='convoy_start_lat',
                                lon_col='convoy_start_lon')
    removed = len(convoy) - len(ocean_convoy)
    print(f"  Removed {removed} mainland convoy origins → {len(ocean_convoy):,} ocean convoys")

    norm  = mcolors.Normalize(vmin=ocean_convoy['n_consecutive_days'].min(),
                              vmax=ocean_convoy['n_consecutive_days'].max())
    cmap  = plt.colormaps['YlOrRd']
    sizes = np.clip(ocean_convoy['n_consecutive_days'] * 4, 12, 150)

    sc = ax.scatter(
        ocean_convoy['convoy_start_lon'], ocean_convoy['convoy_start_lat'],
        c=ocean_convoy['n_consecutive_days'], cmap=cmap, norm=norm,
        s=sizes, alpha=0.8, linewidths=0.25, edgecolors='#444444',
        transform=ccrs.PlateCarree(), zorder=4
    )

    top5 = ocean_convoy.nlargest(5, 'n_consecutive_days')
    for _, row in top5.iterrows():
        label = f"{row['vessel_a'][:12]} &\n{row['vessel_b'][:12]} ({int(row['n_consecutive_days'])}d)"
        ax.annotate(
            label,
            xy=proj.transform_point(row['convoy_start_lon'], row['convoy_start_lat'],
                                    ccrs.PlateCarree()),
            xytext=(8, 4), textcoords='offset points',
            color='#990000', fontsize=6.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none')
        )

    cb = fig.colorbar(sc, ax=ax, orientation='vertical',
                      pad=0.02, shrink=0.65, aspect=25)
    cb.set_label('Consecutive days together', fontsize=10, color='#333333')
    cb.ax.tick_params(labelsize=8, color='#333333')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#333333')

    ax.set_title(
        f'Convoy Start Locations  ·  {len(ocean_convoy):,} convoys  ·  ≥ 3 consecutive shared days',
        fontsize=13, color='#222222', pad=14
    )

    out = os.path.join(OUTPUT_DIR, 'fleet_map_convoys.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out}")


# ── 3. Interactive Folium map ──────────────────────────────────────────────────

def plot_interactive_map(fleet, convoy, land):
    ocean_fleet  = filter_ocean(fleet,  land)
    ocean_convoy = filter_ocean(convoy, land,
                                lat_col='convoy_start_lat',
                                lon_col='convoy_start_lon')

    m = folium.Map(location=[30, -150], zoom_start=3, tiles='CartoDB positron')

    fleet_grp  = folium.FeatureGroup(name='Fleet Events').add_to(m)
    convoy_grp = folium.FeatureGroup(name='Convoy Origins').add_to(m)
    val_grp    = folium.FeatureGroup(name='Spoke-Validated Fleets').add_to(m)

    def fleet_color(n):
        if n >= 7: return '#cc0000'
        if n >= 5: return '#ff7700'
        if n >= 4: return '#ffaa00'
        return '#3377cc'

    for _, row in ocean_fleet.iterrows():
        vessels = json.loads(row['vessel_list']) if isinstance(row['vessel_list'], str) else []
        popup_html = (
            f"<b>Fleet cluster {int(row['cluster_fleet'])}</b><br>"
            f"Vessels: <b>{int(row['n_vessels'])}</b><br>"
            f"Observations: {int(row['n_obs'])}<br>"
            f"Duration: {row['duration_days']:.1f} days<br>"
            f"Leader: {row['leader_voyageID']}<br>"
            f"Spoke-validated: {'✓' if row['validated_by_spoke'] else '✗'}<br>"
            f"Vessels: {', '.join(str(v) for v in vessels[:6])}"
            + ('…' if len(vessels) > 6 else '')
        )
        target = val_grp if row['validated_by_spoke'] else fleet_grp
        folium.CircleMarker(
            location=[row['centroid_lat'], row['centroid_lon']],
            radius=max(4, int(row['n_vessels']) * 1.8),
            color=fleet_color(row['n_vessels']),
            fill=True, fill_opacity=0.7, weight=1,
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(target)

    for _, row in ocean_convoy.iterrows():
        popup_html = (
            f"<b>Convoy</b><br>"
            f"{row['vessel_a']} &amp; {row['vessel_b']}<br>"
            f"Consecutive days: <b>{int(row['n_consecutive_days'])}</b><br>"
            f"Shared clusters: {int(row['n_shared_clusters'])}"
        )
        folium.CircleMarker(
            location=[row['convoy_start_lat'], row['convoy_start_lon']],
            radius=max(3, int(row['n_consecutive_days']) // 2),
            color='#009900', fill=True, fill_opacity=0.5, weight=1,
            popup=folium.Popup(popup_html, max_width=220)
        ).add_to(convoy_grp)

    folium.LayerControl(collapsed=False).add_to(m)

    out = os.path.join(OUTPUT_DIR, 'fleet_map_interactive.html')
    m.save(out)
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading fleet tables from DB...")
    fleet, convoy = load_tables()
    print(f"  fleet_events : {len(fleet):,} rows")
    print(f"  fleet_convoys: {len(convoy):,} rows")

    print("Building land filter...")
    land = build_land_filter()

    print("Plotting fleet event globe map...")
    plot_fleet_map(fleet, land)

    print("Plotting convoy globe map...")
    plot_convoy_map(convoy, land)

    print("Building interactive map...")
    plot_interactive_map(fleet, convoy, land)

    print("\nAll done.")
