#!/usr/bin/env python3

import argparse
import glob
import re
import os
import csv
import numpy as np
from parse_log import extract_run_segments_file
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *
from ovito.data import DataCollection
from multiprocessing import Pool
from tqdm.notebook import tqdm
import time
import pandas as pd
#ovito.enable_logging()
from multiprocessing import Pool, Manager
from filelock import FileLock
import pyarrow as pa
import pyarrow.parquet as pq

def natural_sort_key(path):
    numbers = re.findall(r"\d+", os.path.basename(path))
    return [int(number) for number in numbers] if numbers else [0]


parser = argparse.ArgumentParser(
    description="Perform Wigner-Seitz analysis on OVITO dump files."
)
parser.add_argument("--input-dir", required=True)
parser.add_argument("--output-dir", required=True)

args = parser.parse_args()

input_dir = os.path.abspath(args.input_dir)
output_dir = os.path.abspath(args.output_dir)

os.makedirs(output_dir, exist_ok=True)

if not os.path.isdir(input_dir):
    raise FileNotFoundError(f"Input directory not found: {input_dir}")

dump_files = sorted(
    glob.glob(os.path.join(input_dir, "*.dump")),
    key=natural_sort_key,
)

print(f"Found {len(dump_files)} dump files.")

if not dump_files:
    raise FileNotFoundError(f"No .dump files found in: {input_dir}")

log_files = glob.glob(os.path.join(input_dir, "*.log"))

if not log_files:
    raise FileNotFoundError(f"No *keV.log file found in: {input_dir}")

if len(log_files) > 1:
    raise RuntimeError(
        f"Multiple *keV.log files found in {input_dir}: {log_files}"
    )

logfile = log_files[0]
print(f"Reading log file: {logfile}")

run_segments = extract_run_segments_file(logfile)

if not run_segments:
    raise RuntimeError(f"No run segments found in: {logfile}")

run = run_segments[-1]

step_col = run["col_index"].get("Step")

if step_col is None:
    raise RuntimeError(f"'Step' column not found in: {logfile}")

md_step = np.asarray(run["data"][:, step_col], dtype=np.int64)

#########################
#--- Read dump files ---#
#########################
pipeline = import_file(dump_files, multiple_frames=True)
pipeline.modifiers.append(
    WignerSeitzAnalysisModifier(output_displaced=True,
                                reference_frame=0,
                                affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference)
)
pipeline.modifiers.append(
    DislocationAnalysisModifier(input_crystal_structure=DislocationAnalysisModifier.Lattice.BCC)
)
pipeline.modifiers.append(ExpressionSelectionModifier(expression='Occupancy==1'))
pipeline.modifiers.append(DeleteSelectedModifier())
pipeline.modifiers.append(
    ClusterAnalysisModifier(cutoff=3.5, compute_com=True, compute_gyration=True)
)

#--- Identify loop atoms ---#
def add_cluster_loop_props(frame: int, data: DataCollection):
    cluster_table = data.tables['clusters']
    cluster_ids = data.particles['Cluster']

    # --- Cluster gyration components ---
    gyr = np.array(cluster_table['Gyration Tensor'])
    for i, comp in enumerate(['Gxx','Gxy','Gxz','Gyy','Gyz','Gzz']):
        vals = np.array([gyr[c-1][i] for c in cluster_ids])
        data.particles_.create_property(f'Cluster {comp}', data=vals)

    # --- Cluster COM ---
    coms = np.array([list(v) for v in cluster_table['Center of Mass']])
    for i, ax in enumerate(['X','Y','Z']):
        vals = np.array([coms[c-1][i] for c in cluster_ids])
        data.particles_.create_property(f'Cluster COM {ax}', data=vals)

    # --- Cluster sizes ---
    sizes = np.array(cluster_table['Cluster Size'])
    vals = np.array([sizes[c-1] for c in cluster_ids])
    data.particles_.create_property('Cluster Size', data=vals)

    # --- Loop assignment ---
    loop_flags = np.zeros(len(coms), dtype=bool)
    loop_coms, loop_gyrs = [], []
    for line in data.dislocations.lines:
        pts = np.array(line.points)
        if len(pts) < 3: continue
        com = np.mean(pts, axis=0)
        cen = pts - com
        gyr_t = np.dot(cen.T, cen) / len(pts)
        loop_coms.append(com)
        loop_gyrs.append(gyr_t.flatten()[:6])

    if loop_coms:
        loop_coms, loop_gyrs = np.array(loop_coms), np.array(loop_gyrs)
        for j, lcom in enumerate(loop_coms):
            scores = []
            for i, ccom in enumerate(coms):
                dist = np.linalg.norm(ccom - lcom)
                tensor_diff = np.linalg.norm(gyr[i] - loop_gyrs[j]) / np.linalg.norm(loop_gyrs[j])
                scores.append(dist + tensor_diff)
            loop_flags[np.argmin(scores)] = True

    is_loop = np.array([loop_flags[c-1] for c in cluster_ids], dtype=np.int32)
    data.particles_.create_property('IsLoop', data=is_loop)

pipeline.modifiers.append(add_cluster_loop_props)

#--- Columns to save ---#
props = [
    "Particle Identifier","Particle Type",
    "Position.X","Position.Y","Position.Z","Mass",
    "c_csym","c_potenergy","c_kinenergy",
    "c_disp[1]","c_disp[2]","c_disp[3]","c_disp[4]",
    "Occupancy","Site Type","Site Index","Site Identifier",
    "Structure Type","Color.R","Color.G","Color.B",
    "Cluster","Cluster Gxx","Cluster Gxy","Cluster Gxz",
    "Cluster Gyy","Cluster Gyz","Cluster Gzz",
    "Cluster COM X","Cluster COM Y","Cluster COM Z",
    "Cluster Size","IsLoop"
]

writer = None

output_file = os.path.join(
    output_dir,
    f"{os.path.basename(os.path.normpath(input_dir))}_peratom.dat",
)

print(f"Writing results to: {output_file}")

rows = []
species_types = set()

frames_to_process = [0, pipeline.source.num_frames - 1]
frames_to_process = sorted(set(frames_to_process))

for frame in frames_to_process:
    data = pipeline.compute(frame)

    timestep = int(data.attributes["Timestep"])

    # Access particle properties
    occupancy = np.asarray(data.particles["Occupancy"])
    particle_type = np.asarray(data.particles["Particle Type"])

    is_loop = np.asarray(data.particles["IsLoop"])
    cluster_size = np.asarray(data.particles["Cluster Size"])

    # Interstitial condition
    interstitial_mask = occupancy > 1

    total_interstitials = int(np.count_nonzero(interstitial_mask))

    # Convert values to integer-like values for reliable counting
    interstitial_particle_types = particle_type[interstitial_mask]

    # Count interstitials by particle/species type
    type_counts = {}

    for species in np.unique(interstitial_particle_types):
        species_count = int(
            np.count_nonzero(interstitial_particle_types == species)
        )

        species_name = str(species)
        type_counts[species_name] = species_count
        species_types.add(species_name)

    # Interstitials associated with loops, counted by type
    loop_interstitial_mask = interstitial_mask & (is_loop == 1)
    loop_interstitials = int(np.count_nonzero(loop_interstitial_mask))

    loop_type_counts = {}
    for species in np.unique(particle_type[loop_interstitial_mask]):
        species_name = str(species)
        loop_type_counts[species_name] = int(
            np.count_nonzero(particle_type[loop_interstitial_mask] == species)
        )
        species_types.add(species_name)

    # Interstitials in clusters of size 3 or greater, excluding loops, counted by type
    clustered_nonloop_mask = (
        interstitial_mask
        & (cluster_size >= 3)
        & (is_loop != 1)
    )

    clustered_nonloop_interstitials = int(
        np.count_nonzero(clustered_nonloop_mask)
    )

    clustered_nonloop_type_counts = {}
    for species in np.unique(particle_type[clustered_nonloop_mask]):
        species_name = str(species)
        clustered_nonloop_type_counts[species_name] = int(
            np.count_nonzero(
                particle_type[clustered_nonloop_mask] == species
            )
        )
        species_types.add(species_name)

    row = {
        "Timestep": timestep,
        "tot_ints": total_interstitials,
        "loop_ints": loop_interstitials,
        "cluster_nonloop_ints": clustered_nonloop_interstitials,
        "_type_counts": type_counts,
        "_loop_type_counts": loop_type_counts,
        "_clustered_nonloop_type_counts": clustered_nonloop_type_counts,
    }

    rows.append(row)

    print(
        f"Timestep {timestep}: "
        f"total={total_interstitials}, "
        f"loops={loop_interstitials}, "
        f"non loop clusters>=3={clustered_nonloop_interstitials}"
    )


# Sort species columns for consistent output
species_types = sorted(species_types)

fieldnames = [
    "Timestep",
    "tot_ints",
    "loop_ints",
    "cluster_nonloop_ints",
]

fieldnames.extend(
    f"ints_type_{species}"
    for species in species_types
)

fieldnames.extend(
    f"loop_ints_type_{species}"
    for species in species_types
)

fieldnames.extend(
    f"cluster_nonloop_ints_type_{species}"
    for species in species_types
)

# Write the complete CSV file
with open(output_file, "w", newline="") as results_file:
    writer = csv.DictWriter(results_file, fieldnames=fieldnames)
    writer.writeheader()

    for row in rows:
        output_row = {
            key: row[key]
            for key in fieldnames
            if key in row
        }

        for species in species_types:
            output_row[
                f"ints_type_{species}"
            ] = row["_type_counts"].get(species, 0)
        
        for species in species_types:
            output_row[
                f"loop_ints_type_{species}"
            ] = row["_loop_type_counts"].get(species, 0)

        for species in species_types:
            output_row[
                f"cluster_nonloop_ints_type_{species}"
            ] = row["_clustered_nonloop_type_counts"].get(species, 0)

        writer.writerow(output_row)

print(f"Finished writing results to: {output_file}")

