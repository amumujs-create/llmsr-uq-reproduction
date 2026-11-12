"""Checks for the H2017 robot twin code, on a synthetic 6-joint URDF (no download needed)."""
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis' / 'robot_twin'))
from h2017_dynamics import load_urdf, potential_energy, rnea, with_payload
from saturation import Move, build_cell, torque_min_duration

# Arbitrary non-symmetric geometry and inertias so that couplings are non-trivial.
_JOINTS = [  # origin xyz, origin rpy, axis, mass, com, (ixx iyy izz ixy ixz iyz)
    ('0 0 0.3', '0 0 0', '0 0 1', 12.0, '0.01 0.08 -0.05', (0.30, 0.20, 0.25, 0.01, -0.02, -0.05)),
    ('0 0.02 0', '0 -1.571 -1.571', '0 0 1', 15.0, '0.30 0 0.15', (0.10, 1.50, 1.40, 0.0, 0.05, 0.0)),
    ('0.7 0 0', '0 0 1.571', '0 0 1', 5.0, '0 -0.01 0.04', (0.04, 0.04, 0.03, 0.0, 0.0, -0.002)),
    ('0 -0.6 0', '1.571 0 0', '0 0 1', 4.0, '0 0.10 -0.18', (0.30, 0.28, 0.05, 0.0, 0.0, -0.08)),
    ('0 0 0', '-1.571 0 0', '0 0 1', 2.0, '0 0 0.04', (0.02, 0.02, 0.01, 0.0, 0.0, 0.0)),
    ('0 -0.1 0', '1.571 0 0', '0 0 1', 0.8, '0 0 -0.04', (0.008, 0.007, 0.008, 0.0, 0.0, 0.0)),
]


def _write_urdf(path):
    parts = ['<?xml version="1.0"?><robot name="synthetic"><link name="base"/>']
    parent = 'base'
    for i, (xyz, rpy, axis, m, com, (ixx, iyy, izz, ixy, ixz, iyz)) in enumerate(_JOINTS, 1):
        child = f'link_{i}'
        parts.append(
            f'<joint name="joint_{i}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="{axis}"/>'
            f'<limit effort="500" lower="-3" upper="3" velocity="1.5"/></joint>'
            f'<link name="{child}"><inertial><mass value="{m}"/><origin xyz="{com}"/>'
            f'<inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="{ixy}" ixz="{ixz}" iyz="{iyz}"/>'
            f'</inertial></link>')
        parent = child
    parts.append('</robot>')
    Path(path).write_text(''.join(parts))


class RobotTwinChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / 'synthetic.urdf'
        _write_urdf(path)
        cls.chain = load_urdf(path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_gravity_torque_matches_potential_energy_gradient(self):
        rng = np.random.default_rng(0)
        zero = np.zeros(6)
        for _ in range(5):
            q = rng.uniform(-1.5, 1.5, 6)
            grad = np.array([(potential_energy(self.chain, q + h) - potential_energy(self.chain, q - h)) / 2e-6
                             for h in np.eye(6) * 1e-6])
            np.testing.assert_allclose(rnea(self.chain, q, zero, zero), grad, atol=1e-4)

    def test_mass_matrix_symmetric_positive_definite(self):
        q = np.array([0.3, -0.7, 1.1, 0.4, -0.9, 0.2])
        zero = np.zeros(6)
        mass = np.column_stack([rnea(self.chain, q, zero, e, gravity=0.0) for e in np.eye(6)])
        np.testing.assert_allclose(mass, mass.T, atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(mass).min(), 0)

    def test_velocity_terms_do_no_power(self):
        # qd^T (C(q,qd) qd) = qd^T (Mdot/2) qd  -> check via energy: d/dt KE = qd^T tau when g = 0
        rng = np.random.default_rng(1)
        q, qd, qdd = rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6)
        zero = np.zeros(6)

        def kinetic(qq, vv):
            mass = np.column_stack([rnea(self.chain, qq, zero, e, gravity=0.0) for e in np.eye(6)])
            return 0.5 * vv @ mass @ vv

        dt = 1e-6
        dke = (kinetic(q + qd * dt, qd + qdd * dt) - kinetic(q - qd * dt, qd - qdd * dt)) / (2 * dt)
        self.assertAlmostEqual(dke, qd @ rnea(self.chain, q, qd, qdd, gravity=0.0), places=4)

    def test_payload_raises_static_torque(self):
        q = np.array([0.0, 1.2, 0.3, 0.0, 0.0, 0.0])
        zero = np.zeros(6)
        light = np.abs(rnea(self.chain, q, zero, zero))[1]
        heavy = np.abs(rnea(with_payload(self.chain, 10.0, np.array([0, 0, 0.1])), q, zero, zero))[1]
        self.assertGreater(heavy, light)

    def test_speed_is_linear_below_boundary_and_flat_above(self):
        move = Move('t', (0, 0, 0, 0, 0, 0), (0, 70, 0, 0, 0, 0))
        cell = build_cell(self.chain, move, 0.0)
        s = np.linspace(0.05, 1.0, 20)
        np.testing.assert_allclose(cell.speed(s, regime=False) / s, cell.speed(s, regime=False)[0] / s[0])
        cap = cell.path_deg() / cell.t_torque
        reg = cell.speed(s)
        self.assertTrue(np.all(reg <= cap + 1e-9))
        below = s < cell.boundary
        np.testing.assert_allclose(reg[below], cell.speed(s[below], regime=False))

    def test_torque_min_duration_respects_limits(self):
        move = Move('t', (0, 0, 90, 0, 0, 0), (0, 60, 90, 0, 0, 0))
        limits = np.array([400.0, 400.0, 200.0, 100.0, 60.0, 30.0])
        t = torque_min_duration(self.chain, move, limits)
        self.assertTrue(np.isfinite(t))
        amp = np.deg2rad(np.subtract(move.q1_deg, move.q0_deg))
        q0 = np.deg2rad(move.q0_deg)
        peak = np.zeros(6)
        for tn in np.linspace(0, 1, 401):
            p = 10 * tn**3 - 15 * tn**4 + 6 * tn**5
            dp = (30 * tn**2 - 60 * tn**3 + 30 * tn**4) / t
            ddp = (60 * tn - 180 * tn**2 + 120 * tn**3) / t**2
            peak = np.maximum(peak, np.abs(rnea(self.chain, q0 + amp * p, amp * dp, amp * ddp)))
        self.assertTrue(np.all(peak <= limits * (1 + 1e-3)))
        self.assertGreater(np.max(peak / limits), 0.99)   # tight: some joint touches its limit


class FrictionTwinChecks(unittest.TestCase):
    def setUp(self):
        from friction import FrictionCell, panda_cells, panda_friction
        self.FrictionCell, self.panda_cells, self.panda_friction = FrictionCell, panda_cells, panda_friction

    def test_panda_sigmoid_is_zero_at_rest_and_saturates(self):
        for c in self.panda_cells():
            self.assertAlmostEqual(float(self.panda_friction(0.0, *c.panda)), 0.0, places=12)
            self.assertAlmostEqual(float(c.torque(5.0)), c.fc, places=6)
            self.assertTrue(np.all(np.diff(c.torque(np.linspace(0, 2, 50))) > 0))

    def test_stribeck_reaches_static_friction_at_rest(self):
        c = self.FrictionCell('s', 'stribeck', 1.0, 0.2, fs=1.5, vs=0.05)
        self.assertAlmostEqual(float(c.torque(0.0)), 1.5)
        np.testing.assert_allclose(c.torque(2.0), c.base(2.0), atol=1e-9)

    def test_transition_moves_with_stribeck_speed(self):
        ks = [self.FrictionCell('s', 'stribeck', 1.0, 0.2, fs=1.5, vs=vs).transition() for vs in (0.01, 0.1)]
        self.assertAlmostEqual(ks[1] / ks[0], 10.0, delta=0.3)
        self.assertEqual(self.FrictionCell('n', 'none', 1.0, 0.2).transition(), 0.0)

    def test_regime_and_base_agree_above_transition(self):
        c = self.FrictionCell('s', 'stribeck', 1.0, 0.2, fs=1.6, vs=0.03)
        v = np.geomspace(c.transition() * 2, 2.0, 50)
        self.assertLess(np.max(np.abs(c.torque(v) - c.base(v))), 0.05 * c.fc)


class CollisionTwinChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / 'synthetic.urdf'
        _write_urdf(path)
        cls.chain = load_urdf(path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_no_model_error_gives_zero_residual(self):
        from collision import Scenario, joint_frictions
        fr = joint_frictions('none')
        sc = Scenario(Move('t', (0, 10, 60, 0, 20, 0), (30, 50, 90, 0, 60, 0)), 5.0, 0.0)
        self.assertAlmostEqual(sc.residual_peak(self.chain, 0.8, fr), 0.0, places=9)

    def test_static_residual_scales_with_unregistered_payload(self):
        from collision import Scenario, joint_frictions
        fr = joint_frictions('none')
        move = Move('t', (0, 10, 60, 0, 20, 0), (30, 50, 90, 0, 60, 0))
        y1 = Scenario(move, 5.0, 1.0).residual_peak(self.chain, 1e-4, fr)
        y3 = Scenario(move, 5.0, 3.0).residual_peak(self.chain, 1e-4, fr)
        self.assertAlmostEqual(y3 / y1, 3.0, places=4)


class HoseTwinChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / 'synthetic.urdf'
        _write_urdf(path)
        cls.chain = load_urdf(path)
        cls.pose = (0, 30, 90, 0, 60, 0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_link_frames_agree_with_com_heights(self):
        from h2017_dynamics import com_heights, link_frames
        q = np.array([0.2, -0.4, 0.9, 0.3, -0.7, 1.1])
        z = [(p + r @ l.com)[2] for (p, r), l in zip(link_frames(self.chain, q), self.chain)]
        np.testing.assert_allclose(z, com_heights(self.chain, q), atol=1e-12)

    def test_taut_angle_grows_with_slack_and_shrinks_with_pressure(self):
        from hose import HoseCell
        k_short = HoseCell('a', self.pose, 1.05).taut_angle(self.chain)
        k_long = HoseCell('b', self.pose, 1.20).taut_angle(self.chain)
        k_press = HoseCell('c', self.pose, 1.20, pressure=1.0).taut_angle(self.chain)
        self.assertLess(k_short, k_long)
        self.assertLess(k_press, k_long)   # EPS_PRESSURE < 0: pressurized hose is shorter

    def test_reaction_is_bending_only_well_below_taut_angle(self):
        from hose import HoseCell
        cell = HoseCell('d', self.pose, 1.20)
        k = cell.taut_angle(self.chain)
        a = np.linspace(0, k - 20, 10)
        reg, base = cell.torque(self.chain, a), cell.torque(self.chain, a, regime=False)
        post = abs(cell.torque(self.chain, [k + 30])[0])
        self.assertLess(np.max(np.abs(reg - base)), 0.02 * post)
        self.assertGreater(post, 10 * np.max(np.abs(base)))

    def test_slack_hose_pushes_with_euler_load(self):
        """Deep in slack the axial force is minus the clamped-pinned Euler load, linear in EI."""
        import hose
        cell = hose.HoseCell('e', self.pose, 1.35)
        L = cell.l_eff(self.chain)
        f = cell.tension(0.80 * L, L, cell.e0(self.chain), regime=False)   # e* >> E_C
        p_e = hose.C_EULER * hose.EI_HOSE / L ** 2
        self.assertLess(f, 0.0)
        self.assertAlmostEqual(f / -p_e, 1.0, places=3)
        ei = hose.EI_HOSE
        try:
            hose.EI_HOSE = 2 * ei
            self.assertAlmostEqual(cell.tension(0.80 * L, L, cell.e0(self.chain), regime=False), 2 * f)
        finally:
            hose.EI_HOSE = ei

    def test_taut_point_is_shifted_before_geometric_length(self):
        """Taut length is L (1 - ALPHA e0) < L, and the taut term there is softplus(0)."""
        import hose
        cell = hose.HoseCell('f', self.pose, 1.20)
        L, e0, lt = cell.l_eff(self.chain), cell.e0(self.chain), cell.l_taut(self.chain)
        self.assertGreater(e0, 0.0)
        self.assertAlmostEqual(lt, L * (1 - hose.ALPHA_SHIFT * e0))
        jump = cell.tension(lt, L, e0) - cell.tension(lt, L, e0, regime=False)
        self.assertAlmostEqual(jump, hose.K_TAUT * hose.SOFT_M * np.log(2.0))

    def test_statement_rules_bound_added_error(self):
        """budget_b adds at most b deg of error over R0 for any truth; safe never widens R0."""
        from hose_statements import BUDGETS, apply_rules
        rng = np.random.default_rng(0)
        for _ in range(500):
            k0, ks, kt = rng.uniform(0, 200, 3)
            pred = apply_rules(k0, ks, accepted=bool(rng.integers(2)))
            for b in BUDGETS:
                self.assertLessEqual(abs(pred[f'budget_{b:g}'] - kt) - abs(k0 - kt), b + 1e-9)
            self.assertLessEqual(pred['safe'], k0)

    def test_sweep_recovers_truth_without_noise_or_mismatch(self):
        """Known answer: no noise, exact model, fitted hose length -> a correct statement under R2 is exact."""
        import hose
        from hose_statements_sweep import Model, run_b
        rows = run_b(Model(self.chain), 1, np.random.default_rng(0), sigma=1e-6, estimate_k0=True)
        err = [r[2] for r in rows if r[0] == 'correct' and r[1] == 'R2']
        self.assertTrue(err)
        np.testing.assert_allclose(err, 0.0, atol=1e-9)
        self.assertEqual(hose.EI_HOSE, 0.3)   # Model.get restores the global

    def test_two_coefficient_fit_known_answer(self):
        from hose_gate_mismatch import fit_two
        rng = np.random.default_rng(1)
        A, B = rng.normal(size=40), rng.normal(size=40)
        rss, c = fit_two(2.0 * A + 3.0 * B, A, B)
        np.testing.assert_allclose(c, [2.0, 3.0], atol=1e-10)
        self.assertLess(rss, 1e-20)
        _, c = fit_two(-1.0 * A + 1.0 * B, A, B)    # negative gain is not allowed
        self.assertTrue(np.all(c >= 0))

    def test_calibration_absorbs_gain_and_stiffness(self):
        """Truth = hose.py with torque x1.25 and EI x3: the split A/B fit gives c1 = 1.25*3*EI, c2 = 1.25."""
        import hose
        from hose_gate_mismatch import Basis, fit_gains, fit_two, model_truth
        a0 = np.arange(0.0, 120.0, 2.0)
        y0, _ = model_truth(self.chain, 'j5_60', 1.20, 0.0, a0, dict(scale=1.25, EI_HOSE=3 * hose.EI_HOSE))
        A, B, _ = Basis(self.chain).get('j5_60', 1.20, 0.0, a0)
        for fit in (fit_two, fit_gains):
            rss, c = fit(y0, A, B)
            np.testing.assert_allclose(c, [1.25 * 3 * hose.EI_HOSE, 1.25], rtol=1e-6)
        self.assertEqual(hose.EI_HOSE, 0.3)

    def test_bounded_gain_fit_clips_taut_gain(self):
        from hose_gate_mismatch import GAIN_BOUNDS, fit_gains
        rng = np.random.default_rng(2)
        A, B = np.abs(rng.normal(size=40)), np.abs(rng.normal(size=40))
        _, c = fit_gains(0.3 * A + 5.0 * B, A, B)
        self.assertEqual(c[1], GAIN_BOUNDS[1])
        self.assertGreaterEqual(c[0], 0.0)

    def test_eps_cell_matches_pressure_cell(self):
        """cell_eps(eps) is the full-pressure cell with length change eps; eps=0 gives k0."""
        import hose
        from hose_statements import POSES, cell_eps
        k0 = hose.HoseCell('j5_60', POSES['j5_60'], 1.20, 0.0).taut_angle(self.chain)
        self.assertEqual(cell_eps('j5_60', 1.20, 0.0).taut_angle(self.chain), k0)
        c = cell_eps('j5_60', 1.20, -0.04)
        base = hose.HoseCell('x', POSES['j5_60'], 1.20, 0.0).l_eff(self.chain)
        self.assertAlmostEqual(c.l_eff(self.chain), base * 0.96)


if __name__ == '__main__':
    unittest.main()
