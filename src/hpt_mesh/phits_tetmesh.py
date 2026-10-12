"""
phits_tetmesh.py
------------------------------------------------------------------
Build a PHITS-compatible CONFORMING TETRAHEDRAL mesh of the 5-layer
intestinal block. Unlike the per-layer surface OBJs (which overlap and are
non-manifold after independent smoothing), a single tet mesh is:
  - non-overlapping  (tets tile the volume, share faces)
  - gap-free / conforming (every interior face shared by exactly 2 tets)
  - manifold at all region interfaces (by construction)
so it satisfies PHITS's "no intersections / no non-manifold edges" requirement.

Pipeline:
  1. Outer tissue surface via marching cubes (label>0), keep largest body.
  2. Decimate, then pymeshfix -> guaranteed watertight + manifold + no
     self-intersection PLC boundary.
  3. tetgen fills the interior with quality tets (maxvolume constraint).
  4. Each tet tagged by the LABEL at its centroid (1..5).
  5. Export: tetgen .node/.ele (region in .ele attribute -> PHITS reads this),
     a .vtu for ParaView inspection, and a PHITS [Tetrahedral] snippet.

Region ids: 1=Lumen 2=Mucosa 3=Submucosa 4=Muscularis 5=Serosa
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.measure import marching_cubes
from trimesh.smoothing import filter_taubin
import trimesh
import pymeshfix
import tetgen
import meshio

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_NPZ  = os.path.join(OUT_DIR, "volume_villi.npz")
PHDIR   = os.path.join(OUT_DIR, "phits")
SPACING = (5.0, 8.7616, 8.7616)  # (z, y, x) um  -> z=5um serial sections,
                                 # xy=4.3808 um/px * stride 2. Matches the volume.
SURF_STEP = 3                    # marching-cubes step (native coarsening, no decimation)
SMOOTH_SIGMA = 3.0               # gaussian blur (voxels) on the boundary before meshing
MAXVOL   = 20_000.0              # um^3 per tet. Note the quality mesher already puts
                                 # fine tets in the wall (median edge 22-47 um -> each
                                 # layer is 5-7 tets thick); only the lumen interior is
                                 # coarse. So MAXVOL is not the lever for layer quality
                                 # -- interface smoothness (relayer 3D smoothing) is.
SEROSA_VOX = 3                   # per-slice serosa thickness (voxels) ~30 um at 8.76 um
MIN_DIHEDRAL = 10.0              # deg; suppresses sliver tets
REGION_NAMES = {1: "Lumen", 2: "Mucosa", 3: "Submucosa", 4: "Muscularis", 5: "Serosa"}


def main():
    os.makedirs(PHDIR, exist_ok=True)
    vol = np.load(IN_NPZ)["volume"]
    dz, dy, dx = SPACING
    print(f"volume {vol.shape}, spacing (z,y,x) = {SPACING} um")

    # Serosa: the stored volume has a 3D-dilated serosa that CAPS the top/bottom
    # cut faces. For PHITS those cut faces should be tissue, not serosa. Rebuild
    # serosa as a per-slice (lateral) shell only -> wraps the circumference,
    # leaves the Z-end cut faces as muscularis. (Fragmentation is irrelevant here
    # because tets are labeled by centroid, not by a connected surface.)
    vol = vol.copy()
    vol[vol == 5] = 0                                  # drop 3D-capped serosa
    for i in range(vol.shape[0]):
        tiss = vol[i] > 0
        if not tiss.any():
            continue
        shell = binary_dilation(tiss, iterations=SEROSA_VOX) & ~tiss
        sl = vol[i]; sl[shell] = 5; vol[i] = sl
    print(f"serosa re-lateralized (per-slice, {SEROSA_VOX} vox); "
          f"Z-end cut faces are now tissue")

    # 1) outer surface of ALL tissue (label > 0), gaussian-smoothed so the
    #    boundary is rounded (no voxel staircase / near-contact folds that
    #    break TetGen's boundary recovery)
    occ = gaussian_filter((vol > 0).astype(np.float32), sigma=SMOOTH_SIGMA)
    occ = np.pad(occ, 1)
    verts, faces, _, _ = marching_cubes(occ, level=0.5,
                                        spacing=SPACING, step_size=SURF_STEP)
    verts -= np.array(SPACING)                    # undo the 1-voxel pad
    xyz = verts[:, ::-1]                           # (z,y,x) -> (x,y,z)
    mesh = trimesh.Trimesh(xyz, faces, process=True)
    mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.faces))
    print(f"outer surface: {len(mesh.faces):,} faces (largest body, native coarse)")

    # 2) repair -> clean watertight manifold PLC (no decimation: native coarse
    #    step already reduced faces; decimation is what folded the thin slab)
    v, f = pymeshfix.clean_from_arrays(
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int32))
    # scrub degenerate/duplicate faces & slivers that trip TetGen's recovery
    chk = trimesh.Trimesh(v, f, process=True)
    chk.merge_vertices()
    chk.update_faces(chk.nondegenerate_faces())
    chk.update_faces(chk.unique_faces())
    chk.remove_unreferenced_vertices()
    chk.fix_normals()
    v = np.ascontiguousarray(chk.vertices, dtype=np.float64)
    f = np.ascontiguousarray(chk.faces, dtype=np.int32)
    print(f"repaired PLC: {len(f):,} faces  watertight={chk.is_watertight}  "
          f"winding_ok={chk.is_winding_consistent}  bodies={chk.body_count}")

    # 3) tetrahedralize the interior (nobisect=False -> allow boundary Steiner
    #    points, which lets TetGen recover the constrained faces reliably)
    tet = tetgen.TetGen(v, f)
    # NOTE: tetgen's sliver removal (mindihedral / low minratio) CRASHES natively
    # on this geometry, so we run the stable default quality mesh and drop the
    # few resulting zero-volume slivers in post (below).
    # fixedvolume=True is REQUIRED to activate the maxvolume constraint (tetgen's
    # -a switch). Passing maxvolume alone is silently ignored -- that bug left
    # every prior mesh at the coarse quality-only resolution regardless of MAXVOL.
    tet.tetrahedralize(order=1, quality=True, nobisect=False,
                       fixedvolume=True, maxvolume=MAXVOL)
    nodes = np.asarray(tet.node)                  # (N,3) xyz um
    elems = np.asarray(tet.elem)                  # (M,4) int
    print(f"tetgen: {len(nodes):,} nodes, {len(elems):,} tets")

    # 4) tag each tet by the label at its centroid
    cent = nodes[elems].mean(axis=1)              # (M,3) xyz um
    xi = np.clip((cent[:, 0] / dx).astype(int), 0, vol.shape[2] - 1)
    yi = np.clip((cent[:, 1] / dy).astype(int), 0, vol.shape[1] - 1)
    zi = np.clip((cent[:, 2] / dz).astype(int), 0, vol.shape[0] - 1)
    region = vol[zi, yi, xi].astype(int)
    # boundary tets can land just outside tissue (smoothed/coarsened surface) ->
    # give them the NEAREST tissue label (muscularis at Z-end cut faces, serosa
    # laterally), computed on a downsampled grid to stay memory-safe.
    bad = region == 0
    if bad.any():
        from scipy.ndimage import distance_transform_edt
        ds = 4
        small = vol[::ds, ::ds, ::ds]
        _, (jz, jy, jx) = distance_transform_edt(small == 0, return_indices=True)
        near = small[jz, jy, jx]
        region[bad] = near[np.clip(zi[bad] // ds, 0, near.shape[0] - 1),
                           np.clip(yi[bad] // ds, 0, near.shape[1] - 1),
                           np.clip(xi[bad] // ds, 0, near.shape[2] - 1)]
        region[region == 0] = 4                    # ultra-fallback
        print(f"  {int(bad.sum()):,} boundary tets assigned nearest-tissue label")

    # drop degenerate (zero-volume) sliver tets -> PHITS validity
    p = nodes[elems]
    vol6 = np.einsum('ij,ij->i', np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                     p[:, 3] - p[:, 0])
    keep = np.abs(vol6) > 1e-3
    ndrop = int((~keep).sum())
    elems = elems[keep]
    region = region[keep]
    print(f"dropped {ndrop} zero-volume sliver tets -> {len(elems):,} tets remain")

    # remove nodes orphaned by the sliver drop and reindex (else the QC gate and
    # PHITS see unreferenced nodes)
    used = np.unique(elems)
    if len(used) != len(nodes):
        remap = np.full(len(nodes), -1, np.int64)
        remap[used] = np.arange(len(used))
        nodes = nodes[used]
        elems = remap[elems]
        print(f"  removed {len(remap) - len(used):,} orphaned nodes -> {len(nodes):,}")

    for r in sorted(REGION_NAMES):
        print(f"  region {r} {REGION_NAMES[r]:11s}: {int((region==r).sum()):>9,} tets")

    # ---- HARD QC GATE: refuse to emit a bad mesh (deliverable b) --------------
    from mesh_qc import gate
    gate(nodes, elems, region, raise_on_fail=True)

    # 5a) tetgen .node / .ele (PHITS reads region from the .ele attribute)
    with open(os.path.join(PHDIR, "intestine.node"), "w") as fh:
        fh.write(f"{len(nodes)} 3 0 0\n")
        for i, p in enumerate(nodes):
            fh.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    with open(os.path.join(PHDIR, "intestine.ele"), "w") as fh:
        fh.write(f"{len(elems)} 4 1\n")
        for i, (e, r) in enumerate(zip(elems, region)):
            fh.write(f"{i} {e[0]} {e[1]} {e[2]} {e[3]} {r}\n")

    # 5b) .vtu for ParaView (region as cell data)
    meshio.write_points_cells(
        os.path.join(PHDIR, "intestine.vtu"),
        nodes, [("tetra", elems)],
        cell_data={"region": [region]})

    # 5c) PHITS geometry snippet
    with open(os.path.join(PHDIR, "phits_tet_snippet.txt"), "w") as fh:
        fh.write("[Tetrahedral]\n")
        fh.write("    file = intestine        # reads intestine.node/.ele/.face\n")
        fh.write("    tsf* = 1\n")
        fh.write("    tet_format = tetgen\n")
        fh.write("    scale = 0.1              # um -> mm; set to your PHITS length unit\n\n")
        fh.write("# One PHITS material/region per label (region id = tetgen attribute):\n")
        for r, nm in REGION_NAMES.items():
            fh.write(f"#   region {r} = {nm}\n")

    print(f"\nwrote PHITS mesh to {PHDIR}\\  (intestine.node/.ele/.vtu)")
    print("NOTE: thin serosa (~30 um) is under-resolved at this tet size; "
          "lower MAXVOL to refine (more tets).")


if __name__ == "__main__":
    main()
