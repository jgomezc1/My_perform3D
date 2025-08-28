import importlib.util, sys, pathlib
#from openseespy.opensees import getTime, analyze, test, algorithm, nodeDisp, pattern
#from openseespy.opensees import rayleigh, wipe, getNodeTags, getEleTags, loadConst, timeSeries
from openseespy.opensees import *
import json
import signals as sig
from math import sqrt, pi
import matplotlib.pyplot as plt

def import_explicit(path: pathlib.Path):
    name = "explicit_model_generated"  # avoid clobbering other modules
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def pick_top_master(diaphragms):
    # pick the one with highest Z
    pairs = []
    for rec in diaphragms:
        m = int(rec["master"])
        if m in nodes:
            pairs.append((nodes[m][2], m))  # (z, master_tag)
    pairs.sort(reverse=True)
    return pairs[0][1] if pairs else None

#####
def run_transient_with_tracking(
    npts: int,
    dt: float,
    node_ref: int,
    *,
    tol: float = 1e-8,
    max_iter: int = 10,
    print_every: int = 50,
):
    """
    Runs analyze(1, dt) npts times, tracking:
      - step index (1..npts)
      - OpenSees time via getTime()
      - node_ref displacements (Ux, Uy, Uz)
    If a step fails, tries ModifiedNewton -initial for that step, then restores Newton.
    Returns a dict with arrays for post-processing.
    """
    print("STARTING TIME HISTORY ANALYSIS")
    t_final = npts * dt
    step = 0
    status_codes = []      # 0=OK, nonzero=OpenSees error code
    alg_used = []          # 'Newton' or 'ModifiedNewton-initial'
    times = [getTime()]    # include initial time
    uRef_x = [nodeDisp(node_ref, 1) if step == 0 else 0.0]
    uRef_y = [nodeDisp(node_ref, 2) if step == 0 else 0.0]
    uRef_z = [nodeDisp(node_ref, 3) if step == 0 else 0.0]

    # default solver settings assumed already configured outside this function
    # (constraints, numberer, system, test, algorithm, integrator, analysis)

    while step < npts and getTime() < t_final - 1e-14:
        rc = analyze(1, dt)
        used = "Newton"
        if rc != 0:
            # fallback for this step only
            print(f"[step {step+1}] Newton failed (rc={rc}). Trying ModifiedNewton -initial …")
            test('NormDispIncr', tol, 100, 0)
            algorithm('ModifiedNewton', '-initial')
            rc = analyze(1, dt)
            used = "ModifiedNewton-initial" if rc == 0 else "FAILED"
            # restore defaults
            test('NormDispIncr', tol, max_iter)
            algorithm('Newton')

        step += 1
        t_now = getTime()
        times.append(t_now)
        uRef_x.append(nodeDisp(node_ref, 1))
        uRef_y.append(nodeDisp(node_ref, 2))
        uRef_z.append(nodeDisp(node_ref, 3))
        status_codes.append(rc)
        alg_used.append(used)

        if (step % print_every == 0) or (rc != 0):
            print(f"[step {step:6d}] t={t_now:.6g}  rc={rc}  algo={used}  "
                  f"Ux={uRef_x[-1]:.3e} Uy={uRef_y[-1]:.3e} Uz={uRef_z[-1]:.3e}")

        # optional: early stop on unrecoverable failure
        if rc != 0:
            print(f"[stop] analyze failed even after fallback at step {step}.")
            break

    return {
        "steps_requested": npts,
        "steps_completed": step,
        "dt": dt,
        "times": times,          # length = steps_completed + 1
        "uRef_x": uRef_x,        # same length as times
        "uRef_y": uRef_y,
        "uRef_z": uRef_z,
        "status_codes": status_codes,  # length = steps_completed
        "alg_used": alg_used,          # length = steps_completed
    }


######

repo_root = pathlib.Path().resolve()  # run the notebook from repo root
explicit_path = (repo_root / "out" / "explicit_model.py").resolve()

exp = import_explicit(explicit_path)

wipe()
exp.build_model()  # creates nodes, supports, rigid diaphragms, transforms, elements

print("nodes:", len(getNodeTags() or []))
print("elements:", len(getEleTags() or []))

with open(repo_root / "out" / "diaphragms.json", "r", encoding="utf-8") as f:
    djs = json.load(f)

with open(repo_root / "out" / "nodes.json", "r", encoding="utf-8") as f:
    njs = json.load(f)

nodes = {int(n["tag"]): (float(n["x"]), float(n["y"]), float(n["z"])) for n in njs["nodes"]}
def pick_top_master(diaphragms):
    # pick the one with highest Z
    pairs = []
    for rec in diaphragms:
        m = int(rec["master"])
        if m in nodes:
            pairs.append((nodes[m][2], m))  # (z, master_tag)
    pairs.sort(reverse=True)
    return pairs[0][1] if pairs else None

master = pick_top_master(djs["diaphragms"])
master, nodes.get(master)

ART_DIR = pathlib.Path("out")  # change if your artifacts live elsewhere
dpath = ART_DIR / "diaphragms.json"
npath = ART_DIR / "nodes.json"

# Load artifacts
with dpath.open("r", encoding="utf-8") as f:
    dj = json.load(f)
with npath.open("r", encoding="utf-8") as f:
    nj = json.load(f)

# Map node tag -> z for sorting (and keep x,y if you want)
_nodes = {int(n["tag"]): (float(n["x"]), float(n["y"]), float(n["z"])) for n in nj.get("nodes", [])}

# Collect masters (dedup just in case)
masters_raw = []
story_by_master = {}
for rec in dj.get("diaphragms", []):
    m = int(rec["master"])
    masters_raw.append(m)
    story_by_master[m] = rec.get("story")

masters = sorted(set(masters_raw))  # unique, numeric sort (not elevation)
# Elevation sort (top→bottom by z); keep only masters that exist in nodes.json
masters_top_to_bottom = [m for m, _z in sorted(
    ((m, _nodes[m][2]) for m in masters if m in _nodes),
    key=lambda t: t[1],
    reverse=True
)]

print("Masters (raw order):", masters)
print("Masters (top→bottom by Z):", masters_top_to_bottom)
print("Story map (master_tag → story):", {m: story_by_master.get(m) for m in masters})

# If you just want the list for further use, this is the variable to use:
master_tags = masters_top_to_bottom  # or use `masters` if you don't care about elevation

g   = 9.8
loadConst('-time', 0.0)                        # Set the gravity loads to be constant & reset the time in the domain

dt , npts = sig.morsa_to_opensees("GEN_acc_RotD100RotDarfie2010Wakc_Y.txt", 
                  "DarfieldY.dat", 
                  scale_factor=g, 
                  plot=False)
dt , npts = sig.morsa_to_opensees("GEN_acc_RotD100RotDarfie2010Wakc_X.txt", 
                  "DarfieldX.dat", 
                  scale_factor=g, 
                  plot=False)

print(dt, npts)


timeSeries('Path', 2,  '-filePath', 'DarfieldX.dat', '-dt', 0.02, '-factor', 1.0)  # Set time series to be passed to uniform excitation
timeSeries('Path', 3,  '-filePath', 'DarfieldY.dat', '-dt', 0.02, '-factor', 1.0)
#                            tag , dir ,....,tseriesTag 
pattern('UniformExcitation',  2  , 1, '-accel', 2)                      # Create UniformExcitation load pattern along UX
pattern('UniformExcitation',  3  , 2, '-accel', 3)                      # Create UniformExcitation load pattern along UY
rayleigh(0.1416, 0.0, 0.0, 0.00281)

wipeAnalysis()
tolerance = 1.0e-3
constraints('Transformation')    
numberer('RCM')                                    # Reverse Cuthill-McKee DOF numbering
system('SparseGeneral')                            # Solver for large systems
#test('EnergyIncr', tolerance, 20 , 1)              # Convergence test: energy norm, tolerance, max iterations
#algorithm('ModifiedNewton', '-initial')            # Modified Newton-Raphson algorithm
algorithm('Linear')
integrator('Newmark', 0.5, 0.25)                   # Newmark method (β=0.25, γ=0.5 for constant average)
analysis('Transient')                              # Type of analysis: transient (time history)  


numEigen = 5
eigenValues = eigen(numEigen)
print("eigen values at start of transient:",eigenValues)
for i, lam in enumerate(eigenValues):
    if lam > 0:
        freq = sqrt(lam) / (2 * pi)
        period = 1 / freq
        print(f"Mode {i+1}: Frequency = {freq:.3f} Hz, Period = {period:.3f} s")
    else:
        print(f"Mode {i+1}: Invalid eigenvalue (λ = {lam})")

# --- Example usage (mirrors your loop) ---
# Assumes your analysis stack (constraints('Transformation'), numberer, system, test, algorithm, integrator, analysis)
# is already configured in previous cells.
npts = 4502
dt = 0.02
tolerance = 1e-8
node_ref = 1384010  # your reference node (e.g., diaphragm master)

# Optionally update the default test/algorithm before starting
test('NormDispIncr', tolerance, 10)
algorithm('Newton')

result = run_transient_with_tracking(
    npts=npts,
    dt=dt,
    node_ref=node_ref,
    tol=tolerance,
    max_iter=10,
    print_every=200,  # print every 200 steps (and on failures)
)

# Quick sanity:
print("\n--- Run summary ---")
print("steps completed:", result["steps_completed"], "/", result["steps_requested"])
print("final time     :", result["times"][-1])
print("final Ux,Uy,Uz :", result["uRef_x"][-1], result["uRef_y"][-1], result["uRef_z"][-1])

plt.plot(time, uRef_x)
plt.ylabel('Horizontal Displacement of node 19 (m)-X')
plt.xlabel('Time (s)')
plt.show()