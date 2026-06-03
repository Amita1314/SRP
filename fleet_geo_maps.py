#!/usr/bin/env python3
"""
fleet_geo_maps.py — Geographic visualizations of fleet detection results.

Produces:
  output/fleet_map_static.png   — scatter map of fleet event centroids
  output/fleet_map_convoys.png  — convoy start locations coloured by duration
  output/fleet_map_interactive.html — interactive Folium map
"""
import json
import os
import sqlite3

import folium
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

DB_PATH    = "whaling.db"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_tables():
    con = sqlite3.connect(DB_PATH)
    fleet  = pd.read_sql("SELECT * FROM fleet_events",  con)
    convoy = pd.read_sql("SELECT * FROM fleet_convoys", con)
    con.close()
    return fleet, convoy


# ── 1. Static fleet-event map ──────────────────────────────────────────────────

def plot_fleet_map(fleet):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#0a1628')
    ax.set_facecolor('#0a1628')

    # ocean background grid
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(color='#1a2e4a', linewidth=0.4, linestyle='--')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2e4a')

    # normalise vessel count for colour
    vmin, vmax = 2, fleet['n_vessels'].max()
    norm  = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap  = cm.get_cmap('plasma')

    # size ∝ n_obs (capped)
    sizes = np.clip(fleet['n_obs'] * 1.5, 8, 120)

    sc = ax.scatter(
        fleet['centroid_lon'], fleet['centroid_lat'],
        c=fleet['n_vessels'], cmap=cmap, norm=norm,
        s=sizes, alpha=0.75, linewidths=0.3, edgecolors='white',
        zorder=3
    )

    # highlight top-10 largest fleets
    top10 = fleet.nlargest(10, 'n_vessels')
    ax.scatter(
        top10['centroid_lon'], top10['centroid_lat'],
        s=sizes[top10.index] + 60, facecolors='none',
        edgecolors='#ffcc00', linewidths=1.2, zorder=4
    )
    for _, row in top10.iterrows():
        ax.annotate(
            f"{int(row['n_vessels'])}v",
            xy=(row['centroid_lon'], row['centroid_lat']),
            xytext=(5, 5), textcoords='offset points',
            color='#ffcc00', fontsize=7, fontweight='bold'
        )

    cb = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.01, shrink=0.7)
    cb.set_label('Vessels in fleet', color='white', fontsize=10)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    # validation indicator
    val = fleet[fleet['validated_by_spoke'] == 1]
    if len(val):
        ax.scatter(
            val['centroid_lon'], val['centroid_lat'],
            marker='*', s=40, color='#00ff88', alpha=0.9,
            label=f'Spoke-validated ({len(val)})', zorder=5
        )
        ax.legend(loc='lower left', framealpha=0.3,
                  labelcolor='white', fontsize=9,
                  facecolor='#0a1628', edgecolor='#1a2e4a')

    ax.set_xlabel('Longitude', color='white', fontsize=10)
    ax.set_ylabel('Latitude',  color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    ax.set_title(
        f'Fleet-in-Company Events  |  {len(fleet):,} clusters  |  '
        f'up to {int(fleet["n_vessels"].max())} vessels co-located',
        color='white', fontsize=13, pad=12
    )

    out = os.path.join(OUTPUT_DIR, 'fleet_map_static.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


# ── 2. Convoy start-location map ───────────────────────────────────────────────

def plot_convoy_map(convoy):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#0a1628')
    ax.set_facecolor('#0a1628')

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(color='#1a2e4a', linewidth=0.4, linestyle='--')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2e4a')

    norm = mcolors.Normalize(
        vmin=convoy['n_consecutive_days'].min(),
        vmax=convoy['n_consecutive_days'].max()
    )
    cmap = cm.get_cmap('YlOrRd')
    sizes = np.clip(convoy['n_consecutive_days'] * 4, 12, 150)

    sc = ax.scatter(
        convoy['convoy_start_lon'], convoy['convoy_start_lat'],
        c=convoy['n_consecutive_days'], cmap=cmap, norm=norm,
        s=sizes, alpha=0.8, linewidths=0.3, edgecolors='white', zorder=3
    )

    # annotate longest convoys
    top5 = convoy.nlargest(5, 'n_consecutive_days')
    for _, row in top5.iterrows():
        label = f"{row['vessel_a'][:12]} & {row['vessel_b'][:12]}\n{int(row['n_consecutive_days'])}d"
        ax.annotate(
            label,
            xy=(row['convoy_start_lon'], row['convoy_start_lat']),
            xytext=(8, 4), textcoords='offset points',
            color='#ffcc00', fontsize=6.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='#0a1628', alpha=0.6, ec='none')
        )

    cb = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.01, shrink=0.7)
    cb.set_label('Consecutive days together', color='white', fontsize=10)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    ax.set_xlabel('Longitude', color='white', fontsize=10)
    ax.set_ylabel('Latitude',  color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=8)
    ax.set_title(
        f'Convoy Start Locations  |  {len(convoy):,} convoys  |  '
        f'≥ 3 consecutive shared days',
        color='white', fontsize=13, pad=12
    )

    out = os.path.join(OUTPUT_DIR, 'fleet_map_convoys.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


# ── 3. Interactive Folium map ──────────────────────────────────────────────────

def plot_interactive_map(fleet, convoy):
    m = folium.Map(location=[30, -150], zoom_start=3,
                   tiles='CartoDB dark_matter')

    fleet_grp  = folium.FeatureGroup(name='Fleet Events').add_to(m)
    convoy_grp = folium.FeatureGroup(name='Convoy Origins').add_to(m)
    val_grp    = folium.FeatureGroup(name='Spoke-Validated Fleets').add_to(m)

    # colour fleet events by vessel count
    def fleet_color(n):
        if n >= 7: return '#ff2244'
        if n >= 5: return '#ff8800'
        if n >= 4: return '#ffcc00'
        return '#44aaff'

    for _, row in fleet.iterrows():
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

    # convoy origins
    for _, row in convoy.iterrows():
        popup_html = (
            f"<b>Convoy</b><br>"
            f"{row['vessel_a']} &amp; {row['vessel_b']}<br>"
            f"Consecutive days: <b>{int(row['n_consecutive_days'])}</b><br>"
            f"Shared clusters: {int(row['n_shared_clusters'])}"
        )
        folium.CircleMarker(
            location=[row['convoy_start_lat'], row['convoy_start_lon']],
            radius=max(3, int(row['n_consecutive_days']) // 2),
            color='#00ff88', fill=True, fill_opacity=0.5, weight=1,
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

    print("Plotting fleet event map...")
    plot_fleet_map(fleet)

    print("Plotting convoy map...")
    plot_convoy_map(convoy)

    print("Building interactive map...")
    plot_interactive_map(fleet, convoy)

    print("\nAll done.")
