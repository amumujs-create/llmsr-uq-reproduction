"""Rigid-body dynamics of the Doosan H2017 from its official URDF (numpy only).

The URDF is not vendored. Fetch it at the pinned commit:

    curl -sL https://raw.githubusercontent.com/doosan-robotics/doosan-robot2/\
8653b0440dd23d2b0a57fc413b86269c6967a283/dsr_description2/urdf/h2017.urdf \
        -o data/robot/h2017.urdf

Source: doosan-robotics/doosan-robot2 (BSD-3-Clause), commit 8653b04.
Only <inertial>, <joint origin/axis/limit> are read; meshes are ignored.

The recursive Newton-Euler implementation here was cross-checked once against
MuJoCo 3.2.3 `mj_inverse` (200 random states inside joint limits, max diff
1.1e-7 Nm; 2026-11-05 session). The unit tests check it against an
independent potential-energy finite difference and mass-matrix symmetry.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRAVITY = 9.81
URDF_COMMIT = '8653b0440dd23d2b0a57fc413b86269c6967a283'


def _vec(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()])


def rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis: np.ndarray, q: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * k @ k


@dataclass(frozen=True)
class Link:
    origin_xyz: np.ndarray   # joint origin in parent frame
    origin_rot: np.ndarray   # joint origin rotation (parent <- joint frame at q=0)
    axis: np.ndarray         # joint axis in child frame
    mass: float
    com: np.ndarray          # centre of mass in child frame
    inertia: np.ndarray      # about the COM, child-frame axes
    lower: float
    upper: float
    effort: float
    velocity: float


def load_urdf(path: str | Path) -> list[Link]:
    root = ET.parse(path).getroot()
    links = {l.get('name'): l for l in root.findall('link')}
    chain = []
    for joint in root.findall('joint'):
        if joint.get('type') != 'revolute':
            continue
        origin = joint.find('origin')
        inertial = links[joint.find('child').get('link')].find('inertial')
        if inertial is None:
            raise ValueError(f"link of {joint.get('name')} has no <inertial>")
        att = {k: float(v) for k, v in inertial.find('inertia').attrib.items()}
        inertia = np.array([[att['ixx'], att['ixy'], att['ixz']],
                            [att['ixy'], att['iyy'], att['iyz']],
                            [att['ixz'], att['iyz'], att['izz']]])
        limit = joint.find('limit')
        chain.append(Link(
            origin_xyz=_vec(origin.get('xyz')),
            origin_rot=rpy_to_matrix(*_vec(origin.get('rpy'))),
            axis=_vec(joint.find('axis').get('xyz')),
            mass=float(inertial.find('mass').get('value')),
            com=_vec(inertial.find('origin').get('xyz')),
            inertia=inertia,
            lower=float(limit.get('lower')), upper=float(limit.get('upper')),
            effort=float(limit.get('effort')), velocity=float(limit.get('velocity')),
        ))
    if len(chain) != 6:
        raise ValueError(f'expected 6 revolute joints, found {len(chain)}')
    return chain


def with_payload(chain: list[Link], mass: float, offset: np.ndarray) -> list[Link]:
    """Attach a point mass `mass` at `offset` (flange/link_6 frame) to the last link."""
    if mass <= 0:
        return chain
    last = chain[-1]
    m = last.mass + mass
    com = (last.mass * last.com + mass * offset) / m
    d1, d2 = last.com - com, offset - com
    inertia = (last.inertia
               + last.mass * (d1 @ d1 * np.eye(3) - np.outer(d1, d1))
               + mass * (d2 @ d2 * np.eye(3) - np.outer(d2, d2)))
    new_last = Link(last.origin_xyz, last.origin_rot, last.axis, m, com, inertia,
                    last.lower, last.upper, last.effort, last.velocity)
    return chain[:-1] + [new_last]


def rnea(chain: list[Link], q, qd, qdd, gravity: float = GRAVITY) -> np.ndarray:
    """Joint torques tau = M(q) qdd + C(q, qd) qd + g(q)."""
    q, qd, qdd = (np.asarray(v, dtype=float) for v in (q, qd, qdd))
    n = len(chain)
    w = np.zeros(3); wd = np.zeros(3); a = np.array([0.0, 0.0, gravity])
    rots, forces, moments, coms = [], [], [], []
    for i, link in enumerate(chain):
        rot = link.origin_rot @ axis_angle(link.axis, q[i])
        rt = rot.T
        a = rt @ (a + np.cross(wd, link.origin_xyz) + np.cross(w, np.cross(w, link.origin_xyz)))
        w_parent = rt @ w
        w = w_parent + link.axis * qd[i]
        wd = rt @ wd + np.cross(w_parent, link.axis * qd[i]) + link.axis * qdd[i]
        acc_com = a + np.cross(wd, link.com) + np.cross(w, np.cross(w, link.com))
        rots.append(rot); coms.append(link.com)
        forces.append(link.mass * acc_com)
        moments.append(link.inertia @ wd + np.cross(w, link.inertia @ w))
    tau = np.zeros(n); f = np.zeros(3); m = np.zeros(3)
    for i in reversed(range(n)):
        if i + 1 < n:
            f_child, m_child = rots[i + 1] @ f, rots[i + 1] @ m
            p_child = chain[i + 1].origin_xyz
        else:
            f_child = m_child = p_child = np.zeros(3)
        m = moments[i] + m_child + np.cross(coms[i], forces[i]) + np.cross(p_child, f_child)
        f = forces[i] + f_child
        tau[i] = m @ chain[i].axis
    return tau


def link_frames(chain: list[Link], q) -> list[tuple[np.ndarray, np.ndarray]]:
    """World (position, rotation) of each child link frame."""
    rot = np.eye(3); pos = np.zeros(3); frames = []
    for i, link in enumerate(chain):
        pos = pos + rot @ link.origin_xyz
        rot = rot @ link.origin_rot @ axis_angle(link.axis, q[i])
        frames.append((pos.copy(), rot.copy()))
    return frames


def com_heights(chain: list[Link], q) -> np.ndarray:
    """World z of each link COM (for the potential-energy check)."""
    rot = np.eye(3); pos = np.zeros(3); z = []
    for i, link in enumerate(chain):
        pos = pos + rot @ link.origin_xyz
        rot = rot @ link.origin_rot @ axis_angle(link.axis, q[i])
        z.append((pos + rot @ link.com)[2])
    return np.array(z)


def potential_energy(chain: list[Link], q, gravity: float = GRAVITY) -> float:
    return float(gravity * sum(l.mass * h for l, h in zip(chain, com_heights(chain, q))))
