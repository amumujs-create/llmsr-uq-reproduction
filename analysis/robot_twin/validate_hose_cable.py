"""Check the analytic hose model against a MuJoCo elastic cable (design check, 2026-11-12).

Usage: python validate_hose_cable.py data/robot/h2017.urdf   (needs `mujoco`, tested with 3.2.3)

What is compared (J6 sweep, pressure 0, chord only: hose.R_WRAP = 0):
  (1) fit of the slack force curve used in hose.py (alpha, e_c), with leave-one-pose-out.
  (2) taut angle: pure geometry (l = L), hose.py model (shifted), cable (tension > TAUT_N);
      axial force at q6=0; J6 torque cable / model (chord projection).
  (3) gravity (hose weight) on/off.
  (4) does the slack-window force trend predict k? (linear / quadratic zero crossing)
Not compared: the wrap term (no wrist cylinder in the cable scene) and the slope after
taut (the cable is inextensible; its end force comes from a soft equality constraint).
Hose bending stiffness, mass and diameter are assumptions (sensitivity grid below).
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hose  # noqa: E402
from h2017_dynamics import link_frames, load_urdf  # noqa: E402

POSES = {'j5_30': (0, 30, 90, 0, 30, 0), 'j5_60': (0, 30, 90, 0, 60, 0), 'j5_90': (0, 30, 90, 0, 90, 0)}
SLACKS = [1.05, 1.10, 1.20, 1.35]
RADIUS = 0.0075                      # hose outer radius [m] (1/4" two-wire hose, ~15 mm OD)
EI_GRID = [0.1, 0.3, 1.0]                # bending stiffness [N m^2] (assumption)
MASS_PER_M = 0.4                     # hose mass [kg/m] (assumption)
N_SEG = 25
TAUT_N = 20.0                        # tension that marks 'taut' [N]
STEP_DEG, SETTLE = 0.5, 300          # quasi-static sweep


def endpoints(chain, pose_deg, q6_deg):
    q = np.deg2rad(np.array(pose_deg, dtype=float)); q[5] = np.deg2rad(q6_deg)
    fr = link_frames(chain, q)
    return fr[3][0] + fr[3][1] @ hose.ANCHOR_LINK4, fr[5][0] + fr[5][1] @ hose.TOOL_LINK6


def quat_x_to(d):
    """Quaternion (w,x,y,z) rotating +x onto unit vector d."""
    x = np.array([1.0, 0, 0]); v = np.cross(x, d); c = float(x @ d)
    if np.linalg.norm(v) < 1e-12:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 0, 0, 1])
    q = np.array([1 + c, *v]); return q / np.linalg.norm(q)


def build(A, P0, L0, ei, grav):
    d = (P0 - A) / np.linalg.norm(P0 - A)
    tip = A + L0 * d
    E = ei / (np.pi * RADIUS ** 4 / 4)
    G = E / 2.6
    seg_mass = MASS_PER_M * L0 / N_SEG
    dens = seg_mass / (np.pi * RADIUS ** 2 * (L0 / N_SEG) + 4 / 3 * np.pi * RADIUS ** 3)
    qa = quat_x_to(d)
    xml = f"""
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <option timestep="0.0002" gravity="0 0 {-9.81 if grav else 0}" integrator="implicitfast"/>
  <worldbody>
    <body name="anchor" mocap="true" pos="{A[0]} {A[1]} {A[2]}" quat="{qa[0]} {qa[1]} {qa[2]} {qa[3]}">
      <composite prefix="h" type="cable" curve="s" count="{N_SEG + 1} 1 1" size="{L0}" initial="none">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="{G:.6g}"/><config key="bend" value="{E:.6g}"/><config key="vmax" value="0"/>
        </plugin>
        <joint kind="main" damping="0.02"/>
        <geom type="capsule" size="{RADIUS}" density="{dens:.6g}" contype="0" conaffinity="0"/>
      </composite>
    </body>
    <body name="port" mocap="true" pos="{tip[0]} {tip[1]} {tip[2]}"/>
  </worldbody>
  <equality><connect name="tip" body1="hB_last" body2="port" anchor="{L0 / N_SEG} 0 0" solref="0.001 1"/></equality>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def check(dat, where):
    bad = dat.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
    if bad or not np.all(np.isfinite(dat.qpos)):
        raise FloatingPointError(f'cable sim unstable ({where}, {bad} bad-qacc warnings)')


def tension(dat, d):
    """Signed axial force at the tool end: + pulls (taut), - pushes (bent hose thrust).

    Sign checked by stretching a straight cable 2 mm: the force along anchor->port is +.
    """
    return float(end_force(dat) @ d)


def end_force(dat):
    """Connect-constraint force vector (world); the force on the tool port is its negative."""
    rows = dat.efc_type[:dat.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    return dat.efc_force[:dat.nefc][rows].copy()


def j6_torque(chain, pose, q6_deg, f):
    """Torque about the J6 axis from the hose force on the tool port (Nm)."""
    q = np.deg2rad(np.array(pose, dtype=float)); q[5] = np.deg2rad(q6_deg)
    o6, r6 = link_frames(chain, q)[5]
    P = o6 + r6 @ hose.TOOL_LINK6
    return float(np.cross(P - o6, -f) @ (r6 @ chain[5].axis))


def sweep(chain, pose, slack, ei, grav, k_max):
    """Rows (q6 deg, signed axial force N, J6 torque from the hose Nm)."""
    A, P0 = endpoints(chain, pose, 0.0)
    L0 = slack * np.linalg.norm(P0 - A)
    m = build(A, P0, L0, ei, grav); dat = mujoco.MjData(m)
    pid = m.body('port').mocapid[0]
    start = dat.mocap_pos[pid].copy()
    side = np.cross(P0 - A, [0, 0, 1.0]); side /= np.linalg.norm(side) + 1e-12
    n_ramp = 4000
    for i in range(n_ramp):          # pull the straight cable's tip onto the port, bowing sideways
        s = (i + 1) / n_ramp
        dat.mocap_pos[pid] = (1 - s) * start + s * P0 + np.sin(np.pi * s) * 0.3 * (L0 - np.linalg.norm(P0 - A)) * side
        mujoco.mj_step(m, dat)
    for _ in range(4000):
        mujoco.mj_step(m, dat)
    check(dat, 'setup')
    out = []
    for a in np.arange(0.0, k_max + 40.0, STEP_DEG):
        _, P = endpoints(chain, pose, a)
        dat.mocap_pos[pid] = P
        for _ in range(SETTLE):
            mujoco.mj_step(m, dat)
        check(dat, f'q6={a}')
        t = tension(dat, (P - A) / np.linalg.norm(P - A))
        out.append((a, t, j6_torque(chain, pose, a, end_force(dat))))
        if t > 20 * TAUT_N:
            break
    return np.array(out)


def geom_angle(chain, pose, L, grid=np.arange(0.0, 360.5, 0.5)):
    """Chord-only angle where the required length reaches L (no wrap, no shift)."""
    A = endpoints(chain, pose, 0.0)[0]
    for a in grid:
        if np.linalg.norm(endpoints(chain, pose, a)[1] - A) >= L:
            return float(a)
    return float('inf')


def run_grid(chain):
    """No-gravity sweeps used to fit the slack curve: key (pose, slack, EI) -> (rows, L, l per row)."""
    runs = {}
    for pname, pose in POSES.items():
        A, P0 = endpoints(chain, pose, 0.0)
        for s in SLACKS:
            L = s * np.linalg.norm(P0 - A)
            k = geom_angle(chain, pose, L)
            if k > 200:
                continue
            for ei in EI_GRID:
                r = sweep(chain, pose, s, ei, False, k)
                l = np.array([np.linalg.norm(endpoints(chain, pose, a)[1] - A) for a in r[:, 0]])
                runs[(pname, s, ei)] = (r, L, l, k)
    return runs


def fit_slack(runs, poses):
    """Grid fit of F * L^2 / (C_EULER EI) = -tanh((e - alpha e0) / e_c) on the slack side."""
    E, N, E0 = [], [], []
    for (pname, s, ei), (r, L, l, _) in runs.items():
        if pname in poses:
            e = 1 - l / L
            E.append(e); N.append(r[:, 1] * L ** 2 / (hose.C_EULER * ei)); E0.append(np.full(len(e), e[0]))
    E, N, E0 = map(np.concatenate, (E, N, E0))

    def rms(al, ec):
        m = E > al * E0
        return float(np.sqrt(np.mean((N[m] + np.tanh((E[m] - al * E0[m]) / ec)) ** 2)))
    best = min((rms(al, ec), al, ec) for al in np.arange(0, 0.21, 0.02) for ec in np.arange(0.005, 0.0401, 0.0025))
    return best, rms


def main(urdf):
    chain = load_urdf(urdf)
    print(f'mujoco {mujoco.__version__}; hose radius {RADIUS} m, {MASS_PER_M} kg/m, {N_SEG} segments; '
          f'EI grid {EI_GRID} N m^2; force + = pulls tool, - = pushes')
    runs = run_grid(chain)

    print('\n(1) slack curve fit, F L^2/(C_EULER EI) = -tanh((e - alpha e0)/e_c), rms in Euler-load units')
    (r_all, al, ec), _ = fit_slack(runs, set(POSES))
    print(f'   all poses: alpha {al:.2f}, e_c {ec:.4f}, rms {r_all:.3f}   (hose.py: alpha {hose.ALPHA_SHIFT}, e_c {hose.E_C})')
    for held in POSES:
        (r_tr, a_tr, e_tr), _ = fit_slack(runs, set(POSES) - {held})
        _, rms_te = fit_slack(runs, {held})
        print(f'   hold out {held}: fit alpha {a_tr:.2f} e_c {e_tr:.4f} (train rms {r_tr:.3f}) -> held-out rms {rms_te(a_tr, e_tr):.3f}')

    print('\n(2) hose.py (chord only, EI set per run) vs cable; k_cable = first angle with tension > '
          f'{TAUT_N:.0f} N')
    print(f"   {'pose':6s} {'slack':>5s} {'EI':>4s}  k_geom  k_model  k_cable   F@0 cable  F@0 model  tau6 cable/model")
    R, EI0 = hose.R_WRAP, hose.EI_HOSE
    dk_geom, dk_model, kept = [], [], []
    for (pname, s, ei), (r, L, l, kg) in runs.items():
        hose.R_WRAP, hose.EI_HOSE = 0.0, ei
        cell = hose.HoseCell(pname, POSES[pname], s, 0.0)
        km = cell.taut_angle(chain)
        hit = r[r[:, 1] > TAUT_N]
        kc = hit[0, 0] if len(hit) else float('nan')
        f_model = np.array([cell.tension(li, cell.l_eff(chain), cell.e0(chain)) for li in l])
        tau_model = cell.torque(chain, r[:, 0])
        hose.R_WRAP, hose.EI_HOSE = R, EI0
        slack = r[:, 0] < km - 5
        ratio = np.median(-r[slack, 2] / tau_model[slack])   # cable column is torque applied by the hose
        dk_geom.append(kc - kg); dk_model.append(kc - km); kept.append((pname, s, ei, kc, r))
        print(f'   {pname:6s} {s:5.2f} {ei:4.1f}  {kg:6.1f}  {km:7.1f}  {kc:7.1f}   {r[0, 1]:9.1f}  {f_model[0]:9.1f}  {ratio:6.2f}')
    for name, d in (('k_geom', dk_geom), ('k_model', dk_model)):
        d = np.array(d)
        print(f'   k_cable - {name}: median {np.nanmedian(d):+.1f} deg, range {np.nanmin(d):+.1f} to {np.nanmax(d):+.1f}')

    print('\n(3) gravity on (0.4 kg/m) vs off: shift of k_cable')
    for (pname, s, ei) in [(p, s, e) for p in POSES for s in (1.05, 1.20) for e in (0.1, 1.0)]:
        if (pname, s, ei) not in runs:
            continue
        r0, _, _, kg = runs[(pname, s, ei)]
        rg = sweep(chain, POSES[pname], s, ei, True, kg)
        k0 = r0[r0[:, 1] > TAUT_N][0, 0]; k1 = rg[rg[:, 1] > TAUT_N][0, 0]
        print(f'   {pname} slack {s:.2f} EI {ei:.1f}: off {k0:5.1f}, on {k1:5.1f}, F@0 off {r0[0, 1]:7.1f} on {rg[0, 1]:7.1f}')
    window_trend([x for x in kept if x[2] in (0.1, 1.0)])


def window_trend(kept):
    """Fit the slack-window force and extrapolate to its zero crossing (a data-only guess of k)."""
    print('\n(4) slack-window trend -> guessed k (no gravity); window ends d deg before k_cable, width w')
    errs = {1: [], 2: []}
    for pname, s, ei, kc, r in kept:
        a, f = r[:, 0], r[:, 1]
        for d in (10, 20):
            for w in (20, 40):
                lo, hi = kc - d - w, kc - d
                if lo < 0:
                    continue
                m = (a >= lo) & (a <= hi)
                guess = {}
                for deg in (1, 2):
                    roots = np.roots(np.polyfit(a[m], f[m], deg))
                    ahead = [x.real for x in roots if abs(x.imag) < 1e-9 and x.real > hi]
                    guess[deg] = min(ahead) if ahead else float('inf')
                    errs[deg].append(guess[deg] - kc)
                print(f'   {pname} slack {s:.2f} EI {ei:.1f}: k_cable {kc:5.1f}, window [{lo:5.1f},{hi:5.1f}], '
                      f'F {f[m][0]:7.1f}->{f[m][-1]:7.1f} N, linear guess {guess[1]:6.1f}, quadratic {guess[2]:6.1f}')
    for deg, e in errs.items():
        e = np.array(e)
        print(f'   degree {deg}: guess - k_cable median {np.median(e):+.1f} deg, range {e.min():+.1f} to {e.max():+.1f}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
