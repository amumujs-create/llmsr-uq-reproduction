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


if __name__ == '__main__':
    unittest.main()
