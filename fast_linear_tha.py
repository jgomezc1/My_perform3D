# fast_linear_tha.py  — drop-in replacement for your THA driver
# Key speed-ups:
#  - algorithm('Linear')  (no iterations/tests)
#  - analyze(n_steps, dt) (no Python loop)
#  - consistent dt between Path timeSeries and analyze()
#  - SPD-friendly system + RCM numberer
#  - internal recorders instead of per-step nodeDisp()

import importlib.util, sys, pathlib, json
from math import sqrt, pi
from openseespy.opensees import *
import signals as sig

def import_explicit(path: pathlib.Path):
    name = "explicit_model_generated"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def pick_top_master(diaphragms, nodes_by_tag):
    # Choose the diaphragm master at the highest elevation (z)
    pairs = []
    for rec in diaphragms:
        m = int(rec["master"])
        if m in nodes_by_tag:
            pairs.append((nodes_by_tag[m][2], m))  # (z, tag)
    pairs.sort(reverse=True)
    return pairs[0][1] if pairs else None

# --- Paths ---
repo_root = pathlib.Path().resolve()
out_dir    = (repo_root / "out")
explicit_py = out_dir / "explicit_model.py"

# --- Build model (nodes, elements, rigid diaphragms, etc.) ---
wipe()
exp = import_explicit(explicit_py)
exp.build_model()

# --- Pick a reference node: top diaphragm master ---
with (out_dir / "diaphragms.json").open("r", encoding="utf-8") as f:
    dj = json.load(f)
with (out_dir / "nodes.json").open("r", encoding="utf-8") as f:
    nj = json.load(f)

nodes_by_tag = {int(n["tag"]): (float(n["x"]), float(n["y"]), float(n["z"])) for n in nj.get("nodes", [])}
ref_node = pick_top_master(dj.get("diaphragms", []), nodes_by_tag)

print(f"[INFO] Nodes: {len(getNodeTags() or [])}  Elements: {len(getEleTags() or [])}  Ref node: {ref_node}")

# --- Loads & ground motion (use consistent dt!) ---
g = 9.8
loadConst('-time', 0.0)  # lock-in any gravity state, reset time

# Write both components to .dat using your helper; keep their native dt and npts
dtX, nptsX = sig.morsa_to_opensees("GEN_acc_RotD100RotDarfie2010Wakc_X.txt", "DarfieldX.dat", scale_factor=g, plot=False)
dtY, nptsY = sig.morsa_to_opensees("GEN_acc_RotD100RotDarfie2010Wakc_Y.txt", "DarfieldY.dat", scale_factor=g, plot=False)

if abs(dtX - dtY) > 1e-12:
    raise ValueError(f"X and Y records have different dt (dtX={dtX}, dtY={dtY}). Resample one or use matching records.")
dt = dtX
n_steps = min(nptsX, nptsY)  # run to common length

# Time series and patterns (use the SAME dt as the records)
timeSeries('Path', 2, '-filePath', 'DarfieldX.dat', '-dt', dt, '-factor', 1.0)
timeSeries('Path', 3, '-filePath', 'DarfieldY.dat', '-dt', dt, '-factor', 1.0)
pattern('UniformExcitation', 2, 1, '-accel', 2)  # X
pattern('UniformExcitation', 3, 2, '-accel', 3)  # Y

# Damping (Rayleigh; keep if you need it)
rayleigh(0.1416, 0.0, 0.0, 0.00281)

# --- Analysis setup (FAST path) ---
wipeAnalysis()
constraints('Transformation')
numberer('RCM')

# Prefer SPD solvers for linear frames; fallback is BandGeneral if needed
try:
    system('ProfileSPD')  # often fastest with RCM for building models
except Exception:
    system('BandGeneral')  # robust fallback

# LINEAR algorithm (no iterations, no convergence test)
algorithm('Linear')
integrator('Newmark', 0.5, 0.25)
analysis('Transient')

# --- Recorders (fast, C-side) ---
# Displacements of the reference node (Ux, Uy, Uz)
rec_file = str(out_dir / "ref_node_disp.txt")
recorder('Node', '-file', rec_file, '-time', '-node', ref_node, '-dof', 1, 2, 3, 'disp')

# --- (Optional) Do eigen check separately; comment out for production speed ---
# numEigen = 0  # set >0 to run quick sanity; leave at 0 for max speed
# if numEigen > 0:
#     lam = eigen(numEigen)
#     print("[INFO] First eigenvalue:", lam[0] if lam else None)

# --- Run THA in one shot ---
print(f"[RUN] Linear THA: steps={n_steps}, dt={dt:.6g}, T_final={n_steps*dt:.3f} s")
ok = analyze(n_steps, dt)
if ok != 0:
    print(f"[WARN] analyze returned {ok}. If SPD solver fails, try system('BandGeneral') and re-run.")

# --- Summary ---
t_final = getTime()
print(f"[DONE] Completed {n_steps} steps in {t_final:.3f} s of analysis time.")
print(f"[OUT ] Recorder: {rec_file}")
