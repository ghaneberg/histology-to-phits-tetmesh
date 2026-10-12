"""
phits_tetmesh_gmsh.py
------------------------------------------------------------------
gmsh replacement for phits_tetmesh.py. Same single-conforming-mesh +
per-tet region-tag design, but gmsh's size field ACTUALLY refines
(tetgen's maxvolume won't bind on this geometry and its bgmesh
segfaults). Fine tets in the wall + villi band, coarse in the lumen
center -> thin submucosa/muscularis and the villi are resolved, so the
region boundaries stop pinching to non-manifold edges.

Pipeline:
  1. Outer tissue surface (marching cubes, label>0) -> STL.
  2. gmsh meshes the interior with a size callback: target edge length
     = LC_FINE within FINE_REACH_UM of the wall (covers wall + villi +
     serosa), ramping to LC_COARSE in the deep lumen.
  3. Tag each tet by the label at its centroid.
  4. HARD QC GATE (mesh_qc.gate) -- refuse to emit a bad mesh.
  5. Write tetgen .node/.ele + .vtu + PHITS snippet.

Run:  python phits_tetmesh_gmsh.py [--coarse]   (--coarse = fast uniform test)
"""

import os
import sys
import numpy as np
from scipy.ndimage import (gaussian_filter, binary_dilation,
                           distance_transform_edt)
from skimage.measure import marching_cubes
import trimesh
import pymeshfix
import meshio
import gmsh

from mesh_qc import gate

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_NPZ = os.path.join(OUT_DIR, "volume_villi.npz")
PHDIR = os.path.join(OUT_DIR, "phits")
SPACING = (5.0, 8.7616, 8.7616)     # (z,y,x) um
SMOOTH_SIGMA = 3.0
SURF_STEP = 3
SEROSA_VOX = 7                      # ~61 um synthetic serosa: thin, but >=2 tets at
                                    # LC_FINE so it can mesh as a closed manifold
                                    # (a 30 um serosa is thinner than one tet)

# --- size field (um) ---
LC_FINE   = float(os.environ.get("LC_FINE", "35.0"))
                                    # wall/villi band: submucosa 168um ~5 tets,
                                    # villus 150um ~4 tets
# --- targeted refinement (2026-08-26) ---------------------------------------
# LC_FINE=35 sizes the BROAD bands correctly (submucosa 168um ~4 tets) but the
# fine detail is under-resolved: the stem zone is 40 um deep and crypts are 55 um
# across, versus a 42.7 um median tet edge. Measured consequence: stem-cell volume
# fell 0.653 -> 0.556 mm^3 (-15%) from voxel model to mesh, while broad layers lost
# ~1%. Dose is energy/mass, so that 15% goes straight into the S-value.
# These two knobs refine ONLY crypt/stem and mucosa/villi; submucosa, muscularis
# and serosa keep LC_FINE. Defaults equal LC_FINE, so unset = old behaviour.
LC_STEM  = float(os.environ.get("LC_STEM",  str(LC_FINE)))   # labels 6,7
LC_VILLI = float(os.environ.get("LC_VILLI", str(LC_FINE)))   # label 2
DETAIL_REACH_UM = float(os.environ.get("DETAIL_REACH_UM", "80.0"))  # ramp back to LC_FINE
LC_COARSE = 150.0                   # deep lumen center
FINE_REACH_UM = 450.0               # distance from the wall kept fine (covers the
                                    # ~400um villi projecting into the lumen)
DS_FIELD = 4                        # downsample for the distance field (memory)
REGION_NAMES = {1: "Lumen", 2: "Mucosa", 3: "Submucosa", 4: "Muscularis",
                5: "Serosa", 6: "Crypt", 7: "StemCells"}


def _face_adjacency(elems):
    """Pairs of tets sharing a triangular face."""
    fc = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    M = len(elems)
    faces = np.sort(np.concatenate([elems[:, f] for f in fc]), axis=1)
    tid = np.tile(np.arange(M), 4)
    order = np.lexsort(faces.T)
    fs, ts = faces[order], tid[order]
    same = np.all(fs[1:] == fs[:-1], axis=1)
    return ts[:-1][same], ts[1:][same]


def repair_labels(elems, region, n_pass=4):
    """Remove non-manifold spikes/checkerboards: relabel any tet whose face-
    neighbours nearly all disagree with it, to the majority neighbour region.
    Solid interior tets (neighbours agree with them) are untouched, so real
    layers are preserved; only 1-tet slivers/checkerboard tets flip. This is
    what drives the per-region boundaries toward closed-manifold."""
    a, b = _face_adjacency(elems)
    src = np.concatenate([a, b]); dst = np.concatenate([b, a])
    M = len(elems)
    for _ in range(n_pass):
        counts = np.zeros((M, 8), np.int32)          # region ids 0..7
        np.add.at(counts, (src, region[dst]), 1)
        ntot = counts.sum(1)
        maj = counts.argmax(1)
        majc = counts.max(1)
        flip = (ntot >= 3) & (majc >= 3) & (maj != region)
        if not flip.any():
            break
        region = region.copy()
        region[flip] = maj[flip]
    return region


def reconnect_tissue(elems, region, tissue=(2, 6, 7), fill=2, max_bridge=6):
    """Restore tissue connectivity that CENTROID TAGGING severed.

    `volume_villi.npz` is ONE connected tissue component by construction (add_villi.py
    asserts it). But a villus neck only a few voxels wide can lose its bridging tets to
    the lumen label, leaving the finger floating in the MESH while the MODEL says it is
    attached. So the mesh is what is wrong here, not the model: walk out from the main
    tissue mass through non-tissue tets and relabel the shortest chain that rejoins each
    fragment, rather than deleting tissue the model says is real.

    Bridges longer than `max_bridge` tets are NOT filled -- that is too far to be a
    tagging artifact, and it should reach mesh_qc.gate check [7] and fail loudly."""
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import connected_components
    a, b = _face_adjacency(elems)
    M = len(elems)
    src = np.concatenate([a, b]); dst = np.concatenate([b, a])
    A = csr_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(M, M))
    indptr, indices = A.indptr, A.indices
    region = region.copy()
    total = nfrag = 0
    for _ in range(3):
        tis = np.isin(region, tissue)
        m = tis[a] & tis[b]
        n, lab = connected_components(
            coo_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(M, M)),
            directed=False)
        sizes = np.bincount(lab[tis], minlength=n)
        if int((sizes > 0).sum()) <= 1:
            break
        main = int(sizes.argmax())
        # BFS outward from the main tissue mass, through NON-tissue tets only
        depth = np.full(M, -1, np.int32)
        cur = tis & (lab == main)
        depth[cur] = 0
        for d in range(1, max_bridge + 1):
            nb = (A @ cur.astype(np.int32)) > 0
            nb &= (depth < 0) & ~tis
            if not nb.any():
                break
            depth[nb] = d
            cur = nb
        fixed = 0
        for c in np.flatnonzero(sizes > 0):
            if c == main:
                continue
            frag = np.flatnonzero(tis & (lab == c))
            cand = np.unique(np.concatenate(
                [indices[indptr[t]:indptr[t + 1]] for t in frag]))
            cand = cand[depth[cand] > 0]
            if len(cand) == 0:
                continue                      # no short bridge -> leave for the gate
            t = int(cand[np.argmin(depth[cand])])
            while depth[t] > 0:
                region[t] = fill
                total += 1
                nbrs = indices[indptr[t]:indptr[t + 1]]
                nxt = nbrs[depth[nbrs] == depth[t] - 1]
                if len(nxt) == 0:
                    break
                t = int(nxt[0])
            fixed += 1
        nfrag += fixed
        if fixed == 0:
            break
    print(f"  reconnect_tissue: rejoined {nfrag:,} fragment(s) by relabelling "
          f"{total:,} bridge tets -> region {fill}")
    return region


def build_surface(vol):
    """Outer tissue surface (label>0), watertight manifold, um coords."""
    occ = gaussian_filter((vol > 0).astype(np.float32), sigma=SMOOTH_SIGMA)
    occ = np.pad(occ, 1)
    verts, faces, _, _ = marching_cubes(occ, level=0.5, spacing=SPACING,
                                        step_size=SURF_STEP)
    verts -= np.array(SPACING)
    xyz = verts[:, ::-1]                              # (z,y,x)->(x,y,z)
    m = trimesh.Trimesh(xyz, faces, process=True)
    m = max(m.split(only_watertight=False), key=lambda x: len(x.faces))
    v, f = pymeshfix.clean_from_arrays(
        np.ascontiguousarray(m.vertices, np.float64),
        np.ascontiguousarray(m.faces, np.int32))
    chk = trimesh.Trimesh(v, f, process=True)
    chk.merge_vertices(); chk.update_faces(chk.nondegenerate_faces())
    chk.update_faces(chk.unique_faces()); chk.remove_unreferenced_vertices()
    chk.fix_normals()
    print(f"outer surface: {len(chk.faces):,} faces  watertight={chk.is_watertight}")
    return chk


def main():
    coarse = "--coarse" in sys.argv
    os.makedirs(PHDIR, exist_ok=True)
    vol = np.load(IN_NPZ)["volume"].copy()
    dz, dy, dx = SPACING
    print(f"volume {vol.shape}, spacing {SPACING} um  (coarse={coarse})")

    # re-lateralize serosa (per-slice shell; Z-end cut faces stay tissue)
    vol[vol == 5] = 0
    for i in range(vol.shape[0]):
        t = vol[i] > 0
        if t.any():
            vol[i][binary_dilation(t, iterations=SEROSA_VOX) & ~t] = 5

    surf = build_surface(vol)
    stl = os.path.join(PHDIR, "_outer.stl")
    surf.export(stl)

    # distance-to-wall field (um) on a downsampled grid -> drives the size field
    wall = np.isin(vol, (2, 3, 4, 6, 7))               # crypts+stem are fine detail
                                                       # -> keep them in the fine zone
    samp = (dz*DS_FIELD, dy*DS_FIELD, dx*DS_FIELD)
    d_wall = distance_transform_edt(~wall[::DS_FIELD, ::DS_FIELD, ::DS_FIELD],
                                    sampling=samp)
    fz, fy, fx = d_wall.shape

    # Extra distance fields so the size can key on WHAT the tissue is, not just how
    # far it is from the wall. Skipped entirely when the knobs are at their defaults.
    detail = LC_STEM < LC_FINE or LC_VILLI < LC_FINE
    if detail:
        d_stem = distance_transform_edt(
            ~np.isin(vol, (6, 7))[::DS_FIELD, ::DS_FIELD, ::DS_FIELD], sampling=samp)
        d_vil = distance_transform_edt(
            (vol != 2)[::DS_FIELD, ::DS_FIELD, ::DS_FIELD], sampling=samp)
        print(f"  targeted refinement: LC_STEM={LC_STEM} LC_VILLI={LC_VILLI} "
              f"LC_FINE={LC_FINE} reach={DETAIL_REACH_UM} um")

    def _ramp(d, lc_in, lc_out, reach):
        if d <= 0:
            return lc_in
        if d >= reach:
            return lc_out
        return lc_in + (lc_out - lc_in) * (d / reach)

    def size_at(x, y, z):
        iz = min(max(int(z/(dz*DS_FIELD)), 0), fz-1)
        iy = min(max(int(y/(dy*DS_FIELD)), 0), fy-1)
        ix = min(max(int(x/(dx*DS_FIELD)), 0), fx-1)
        d = d_wall[iz, iy, ix]
        if d <= 0:
            out = LC_FINE
        elif d >= FINE_REACH_UM:
            out = LC_COARSE
        else:
            out = LC_FINE + (LC_COARSE - LC_FINE) * (d / FINE_REACH_UM)
        if detail:
            # Ramp back to `out` (the wall-based size), NOT to LC_FINE. Ramping to
            # LC_FINE clamps the whole domain at 35 um -- it dragged the deep lumen
            # down from LC_COARSE=150 and put ~5.8M tets somewhere dose does not care.
            out = min(out, _ramp(d_stem[iz, iy, ix], LC_STEM, out, DETAIL_REACH_UM))
            out = min(out, _ramp(d_vil[iz, iy, ix], LC_VILLI, out, DETAIL_REACH_UM))
        return out

    # ---- gmsh: mesh the interior of the STL with the size field ----
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    # tolerate the tiny near-coplanar facet pairs marching cubes produces
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.001)
    gmsh.merge(stl)
    # Use the STL triangulation AS the boundary mesh (do NOT classifySurfaces --
    # that remeshes the surface and creates overlapping facets). Fill interior only.
    s = gmsh.model.getEntities(2)
    sl = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    if coarse:
        gmsh.option.setNumber("Mesh.MeshSizeMin", LC_COARSE)
        gmsh.option.setNumber("Mesh.MeshSizeMax", LC_COARSE)
    else:
        gmsh.model.mesh.setSizeCallback(
            lambda dim, tag, x, y, z, lc: size_at(x, y, z))
        # MUST track the finest tier. Left at LC_FINE this silently floors every
        # refinement request: LC_STEM=22 and LC_STEM=15 both clamped to 35 and
        # produced byte-identical meshes (2026-08-26).
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(LC_FINE, LC_STEM, LC_VILLI))
        gmsh.option.setNumber("Mesh.MeshSizeMax", LC_COARSE)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)       # Delaunay
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)    # Netgen opt HANGS on this
                                                       # geometry (stuck on illegal
                                                       # tets); base + gate instead
    gmsh.option.setNumber("Mesh.Optimize", 1)          # cheap Laplacian/relocate
    print("meshing with gmsh ...")
    gmsh.model.mesh.generate(3)

    ntags, ncoords, _ = gmsh.model.mesh.getNodes()
    nodes = ncoords.reshape(-1, 3)
    id2idx = {int(t): i for i, t in enumerate(ntags)}
    etypes, etags, enodes = gmsh.model.mesh.getElements(3)
    tets = enodes[0].reshape(-1, 4)
    elems = np.vectorize(id2idx.get)(tets)
    gmsh.finalize()
    print(f"gmsh: {len(nodes):,} nodes, {len(elems):,} tets")

    # ---- tag by centroid label ----
    cent = nodes[elems].mean(axis=1)
    xi = np.clip((cent[:, 0]/dx).astype(int), 0, vol.shape[2]-1)
    yi = np.clip((cent[:, 1]/dy).astype(int), 0, vol.shape[1]-1)
    zi = np.clip((cent[:, 2]/dz).astype(int), 0, vol.shape[0]-1)
    region = vol[zi, yi, xi].astype(int)
    bad = region == 0
    if bad.any():
        ds = 4
        small = vol[::ds, ::ds, ::ds]
        _, (jz, jy, jx) = distance_transform_edt(small == 0, return_indices=True)
        near = small[jz, jy, jx]
        region[bad] = near[np.clip(zi[bad]//ds, 0, near.shape[0]-1),
                           np.clip(yi[bad]//ds, 0, near.shape[1]-1),
                           np.clip(xi[bad]//ds, 0, near.shape[2]-1)]
        region[region == 0] = 4
        print(f"  {int(bad.sum()):,} boundary tets -> nearest-tissue label")

    # repair non-manifold spikes/checkerboards -> closed-manifold region boundaries
    before = region.copy()
    region = repair_labels(elems, region)
    print(f"  label repair: {int((region != before).sum()):,} tets relabeled")

    # fix inverted tets (negative volume = orientation only) by swapping 2 nodes,
    # then drop truly-degenerate ones + reindex orphaned nodes
    p = nodes[elems]
    v6 = np.einsum('ij,ij->i', np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]),
                   p[:, 3]-p[:, 0])
    inv = v6 < 0
    if inv.any():
        elems[inv] = elems[inv][:, [0, 1, 3, 2]]       # swap -> positive volume
        print(f"  fixed {int(inv.sum()):,} inverted tets (node swap)")
    keep = np.abs(v6) > 1e-3
    elems, region = elems[keep], region[keep]
    used = np.unique(elems)
    if len(used) != len(nodes):
        remap = np.full(len(nodes), -1, np.int64); remap[used] = np.arange(len(used))
        nodes = nodes[used]; elems = remap[elems]
    # centroid tagging can sever thin villus necks the voxel model has connected
    region = reconnect_tissue(elems, region)
    print(f"kept {len(elems):,} tets, {len(nodes):,} nodes")
    for r in sorted(REGION_NAMES):
        print(f"  region {r} {REGION_NAMES[r]:11s}: {int((region==r).sum()):>10,} tets")

    # ---- HARD QC GATE ----
    gate(nodes, elems, region, raise_on_fail=True)

    # ---- write ----
    with open(os.path.join(PHDIR, "intestine.node"), "w") as fh:
        fh.write(f"{len(nodes)} 3 0 0\n")
        for i, q in enumerate(nodes):
            fh.write(f"{i} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n")
    with open(os.path.join(PHDIR, "intestine.ele"), "w") as fh:
        fh.write(f"{len(elems)} 4 1\n")
        for i, (e, r) in enumerate(zip(elems, region)):
            fh.write(f"{i} {e[0]} {e[1]} {e[2]} {e[3]} {r}\n")
    meshio.write_points_cells(os.path.join(PHDIR, "intestine.vtu"),
                              nodes, [("tetra", elems)],
                              cell_data={"region": [region]})
    print(f"\nwrote PHITS mesh (gmsh) to {PHDIR}")


if __name__ == "__main__":
    main()
