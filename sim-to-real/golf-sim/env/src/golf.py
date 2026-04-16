from pathlib import Path
import sys
import numpy as np
import mujoco

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
from course import (
    generate_course, HOLE_R, BOUNCE_RESTITUTION,
    WALL_H as _DEFAULT_WALL_H, WALL_T as _DEFAULT_WALL_T,
)

BALL_RADIUS = 0.02135
BALL_MASS = 0.04593
GRAVITY = 9.81
CONTACT_NUDGE = 1e-6
STEP_MAX_BOUNCES = 8
BOUNDARY_HALF_EXTENT = 2.5
BOUNDARY_WALL_HALF_THICKNESS = 0.01

DEFAULT_PARAMS = {
    "ground_friction": [0.5, 0.005, 0.01],
    "ground_solref": [0.02, 1.0],
    "ball_mass": BALL_MASS,
}

_BASE_XML = (_here / "golf.xml").read_text()


def _build_xml(spec):
    wall_h = spec.get("wall_h", _DEFAULT_WALL_H)
    wall_t = spec.get("wall_t", _DEFAULT_WALL_T)
    lines = []
    for i, (pos, angle, hl) in enumerate(spec["walls"]):
        if hl <= 0:
            continue
        p1, p2 = _wall_endpoints((pos, angle, hl))
        lines.append(
            f'    <geom name="obs_{i}" type="capsule" '
            f'fromto="{p1[0]:.4f} {p1[1]:.4f} {wall_h / 2:.4f} '
            f'{p2[0]:.4f} {p2[1]:.4f} {wall_h / 2:.4f}" '
            f'size="{wall_t:.4f}" rgba=".55 .35 .18 1" '
            f'contype="0" conaffinity="0"/>'
        )
    for i, (pos, radius) in enumerate(spec["posts"]):
        lines.append(
            f'    <geom name="post_{i}" type="cylinder" '
            f'pos="{pos[0]:.4f} {pos[1]:.4f} {wall_h / 2:.4f}" '
            f'size="{radius:.4f} {wall_h / 2:.4f}" rgba=".6 .6 .6 1" '
            f'contype="0" conaffinity="0"/>'
        )
    h = spec["hole"]
    lines.append(
        f'    <geom name="hole" type="cylinder" '
        f'pos="{h[0]:.4f} {h[1]:.4f} 0.001" '
        f'size="{HOLE_R} 0.001" rgba="1 1 1 1" '
        f'contype="0" conaffinity="0"/>'
    )
    return _BASE_XML.replace("</worldbody>", "\n".join(lines) + "\n  </worldbody>")


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _wall_endpoints(wall):
    pos, angle_deg, hl = wall
    angle = np.radians(angle_deg)
    delta = hl * np.array([np.cos(angle), np.sin(angle)])
    return pos - delta, pos + delta


def _travel_time(speed, decel, dist):
    if dist <= 0:
        return 0.0
    if decel <= 1e-12:
        return dist / max(speed, 1e-12)
    disc = max(speed * speed - 2 * decel * dist, 0.0)
    return (speed - np.sqrt(disc)) / decel


def _closest_point_on_segment(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom <= 1e-12:
        return a.copy()
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    return a + t * ab


class GolfEnv:

    def __init__(self, params=None, seed=None, n_bounces=None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.new_course(seed, n_bounces=n_bounces)

    def _init_model(self, xml):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self._ground_id = self._geom_id("ground")
        self._ball_geom_id = self._geom_id("ball_geom")
        self._ball_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        joint_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        self._qpos_adr = self.model.jnt_qposadr[joint_x]
        self._qvel_adr = self.model.jnt_dofadr[joint_x]

        self._apply_params()

    def _geom_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def _apply_params(self):
        self.model.body_mass[self._ball_body_id] = self.params["ball_mass"]

    def _linear_decel(self):
        return max(1e-6, self.params["ground_friction"][0] * GRAVITY)

    def _bounce_restitution(self):
        timeconst, dampratio = self.params["ground_solref"]
        tc_scale = np.clip(np.sqrt(0.02 / max(timeconst, 1e-6)), 0.85, 1.15)
        dr_scale = np.clip(1.0 - 0.05 * (dampratio - 1.0), 0.9, 1.1)
        return float(np.clip(BOUNCE_RESTITUTION * tc_scale * dr_scale, 0.5, 0.999))

    def _path_nodes(self):
        return (
            [self.ball_start]
            + [np.array(w, dtype=float) for w in self.waypoints[1:-1]]
            + [self.hole_pos]
        )

    def _path_segments(self):
        nodes = self._path_nodes()
        return list(zip(nodes[:-1], nodes[1:]))

    def _compute_solution_velocity(self):
        path = self._path_nodes()
        speed_needed = 0.0
        bounce_loss = self._bounce_restitution()
        decel = self._linear_decel()

        for seg_idx in range(len(path) - 2, -1, -1):
            seg_len = np.linalg.norm(path[seg_idx + 1] - path[seg_idx])
            if seg_idx < len(path) - 2:
                speed_needed /= bounce_loss
            speed_needed = np.sqrt(speed_needed * speed_needed + 2 * decel * seg_len)

        direction = path[1] - path[0]
        direction /= np.linalg.norm(direction)
        return (direction * speed_needed).tolist()

    def _boundary_segments(self):
        bound = BOUNDARY_HALF_EXTENT - BOUNDARY_WALL_HALF_THICKNESS - BALL_RADIUS
        return [
            (np.array([bound, -bound]), np.array([bound, bound])),
            (np.array([-bound, -bound]), np.array([-bound, bound])),
            (np.array([-bound, bound]), np.array([bound, bound])),
            (np.array([-bound, -bound]), np.array([bound, -bound])),
        ]

    def new_course(self, seed=None, n_bounces=None):
        for attempt in range(20):
            s = None if seed is None else seed + attempt
            spec = generate_course(s, n_bounces=n_bounces)
            self.ball_start = np.array(spec["start"], dtype=float)
            self.hole_pos = np.array(spec["hole"], dtype=float)
            self.waypoints = [np.array(w, dtype=float) for w in spec["waypoints"]]
            self.n_bounces = spec["n_bounces"]
            self._wall_segments = [
                (np.array(a, dtype=float), np.array(b, dtype=float))
                for a, b in spec["collision_walls"]
                if np.linalg.norm(np.array(b) - np.array(a)) > 1e-12
            ]
            self._wall_segments.extend(self._boundary_segments())
            wall_t = spec.get("wall_t", _DEFAULT_WALL_T)
            self._wall_capsules = []
            for wall in spec["obstacle_walls"]:
                a, b = _wall_endpoints(wall)
                if np.linalg.norm(b - a) <= 1e-12:
                    continue
                self._wall_capsules.append((a, b, wall_t + BALL_RADIUS))
            self._posts = [
                (np.array(pos, dtype=float), radius + BALL_RADIUS)
                for pos, radius in spec["posts"]
            ]
            self._init_model(_build_xml(spec))
            self.solution_velocity = self._compute_solution_velocity()
            self.reset_ball()
            if np.all(np.abs(self.hole_pos) < 2.3):
                return
        raise RuntimeError("failed to build a valid course")

    def reset_ball(self, pos=None):
        if pos is None:
            pos = self.ball_start
        self.data.time = 0.0
        self.data.qpos[self._qpos_adr:self._qpos_adr + 2] = pos
        self.data.qvel[self._qvel_adr:self._qvel_adr + 2] = 0
        mujoco.mj_forward(self.model, self.data)

    def set_velocity(self, vx, vy):
        self.data.qvel[self._qvel_adr:self._qvel_adr + 2] = [vx, vy]
        mujoco.mj_forward(self.model, self.data)

    def _segment_hit(self, pos, move, a, b):
        denom = _cross2(move, b - a)
        if abs(denom) < 1e-12:
            return None

        delta = a - pos
        t = _cross2(delta, b - a) / denom
        u = _cross2(delta, move) / denom
        if not (1e-9 < t <= 1.0 and 0.0 <= u <= 1.0):
            return None

        tangent = b - a
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        if np.dot(move, normal) > 0:
            normal = -normal
        return {
            "dist": t * np.linalg.norm(move),
            "point": pos + t * move,
            "normal": normal,
        }

    def _wall_hit(self, pos, move, a, b, radius):
        move_len = np.linalg.norm(move)
        if move_len < 1e-12:
            return None

        best = None
        tangent = b - a
        seg_len = np.linalg.norm(tangent)
        if seg_len > 1e-12:
            tangent /= seg_len
            normal = np.array([-tangent[1], tangent[0]])
            denom = np.dot(move, normal)
            if abs(denom) > 1e-12:
                offset0 = np.dot(pos - a, normal)
                for side in (-1.0, 1.0):
                    tau = (side * radius - offset0) / denom
                    if not (1e-9 < tau <= 1.0):
                        continue
                    hit_point = pos + tau * move
                    proj = np.dot(hit_point - a, tangent)
                    if not (0.0 <= proj <= seg_len):
                        continue
                    wall_point = a + proj * tangent
                    hit_normal = hit_point - wall_point
                    norm = np.linalg.norm(hit_normal)
                    if norm < 1e-12:
                        continue
                    hit_normal /= norm
                    if np.dot(move, hit_normal) >= -1e-12:
                        continue
                    hit = {
                        "dist": tau * move_len,
                        "point": hit_point,
                        "normal": hit_normal,
                    }
                    if best is None or hit["dist"] < best["dist"]:
                        best = hit

        for center in (a, b):
            hit = self._post_hit(pos, move, center, radius)
            if hit is not None and (best is None or hit["dist"] < best["dist"]):
                best = hit

        return best

    def _post_hit(self, pos, move, center, radius):
        move_len = np.linalg.norm(move)
        if move_len < 1e-12:
            return None

        direction = move / move_len
        rel = pos - center
        b = 2 * np.dot(direction, rel)
        c = np.dot(rel, rel) - radius * radius
        disc = b * b - 4 * c
        if disc < 0:
            return None

        sqrt_disc = np.sqrt(disc)
        best = None
        for root in ((-b - sqrt_disc) / 2, (-b + sqrt_disc) / 2):
            if 1e-9 < root <= move_len:
                hit_point = pos + root * direction
                normal = hit_point - center
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                if np.dot(direction, normal) > 0:
                    normal = -normal
                hit = {"dist": root, "point": hit_point, "normal": normal}
                if best is None or hit["dist"] < best["dist"]:
                    best = hit
        return best

    def _first_collision(self, pos, move):
        best = None
        for a, b in self._wall_segments:
            hit = self._segment_hit(pos, move, a, b)
            if hit is not None and (best is None or hit["dist"] < best["dist"]):
                best = hit
        for a, b, radius in self._wall_capsules:
            hit = self._wall_hit(pos, move, a, b, radius)
            if hit is not None and (best is None or hit["dist"] < best["dist"]):
                best = hit
        for center, radius in self._posts:
            hit = self._post_hit(pos, move, center, radius)
            if hit is not None and (best is None or hit["dist"] < best["dist"]):
                best = hit
        return best

    def step(self):
        qa = self._qpos_adr
        va = self._qvel_adr
        remaining = self.model.opt.timestep
        decel = self._linear_decel()
        restitution = self._bounce_restitution()

        for _ in range(STEP_MAX_BOUNCES):
            vel = self.data.qvel[va:va + 2].copy()
            speed = np.linalg.norm(vel)
            if speed <= 1e-12:
                self.data.qvel[va:va + 2] = 0
                self.data.time += remaining
                break

            step_time = min(remaining, speed / decel)
            distance = speed * step_time - 0.5 * decel * step_time * step_time
            if distance <= 1e-12:
                self.data.qvel[va:va + 2] = 0
                self.data.time += remaining
                break

            direction = vel / speed
            move = direction * distance
            hit = self._first_collision(self.data.qpos[qa:qa + 2], move)

            if hit is None:
                self.data.qpos[qa:qa + 2] += move
                new_speed = max(0.0, speed - decel * step_time)
                self.data.qvel[va:va + 2] = direction * new_speed
                self.data.time += step_time
                remaining -= step_time
                if remaining <= 1e-12:
                    break
                continue

            hit_time = _travel_time(speed, decel, hit["dist"])
            speed_at_hit = max(0.0, speed - decel * hit_time)
            self.data.qpos[qa:qa + 2] = hit["point"] + CONTACT_NUDGE * hit["normal"]

            vel_at_hit = direction * speed_at_hit
            vn = np.dot(vel_at_hit, hit["normal"])
            reflected = vel_at_hit - 2.0 * vn * hit["normal"]
            reflected *= restitution
            self.data.qvel[va:va + 2] = reflected

            self.data.time += hit_time
            remaining -= hit_time
            if remaining <= 1e-12:
                break

        mujoco.mj_forward(self.model, self.data)

    def solve(self):
        return self.solution_velocity

    def rollout(self, initial_velocity, max_time=10.0, vel_threshold=1e-4,
                dt_record=0.01):
        self.reset_ball()
        self.set_velocity(*initial_velocity)

        qa = self._qpos_adr
        va = self._qvel_adr
        times, positions, velocities = [], [], []
        next_record = 0.0

        while self.data.time < max_time:
            if self.data.time >= next_record:
                times.append(self.data.time)
                positions.append(np.r_[self.data.qpos[qa:qa + 2], BALL_RADIUS])
                velocities.append(np.r_[self.data.qvel[va:va + 2], 0.0])
                next_record += dt_record

            self.step()

            speed = np.linalg.norm(self.data.qvel[va:va + 2])
            if speed < vel_threshold and self.data.time > 0.05:
                times.append(self.data.time)
                positions.append(np.r_[self.data.qpos[qa:qa + 2], BALL_RADIUS])
                velocities.append(np.r_[self.data.qvel[va:va + 2], 0.0])
                break

        return {
            "times": np.array(times),
            "positions": np.array(positions),
            "velocities": np.array(velocities),
        }

    def ball_in_hole(self, pos=None):
        if pos is None:
            pos = self.data.qpos[self._qpos_adr:self._qpos_adr + 2]
        return np.linalg.norm(pos[:2] - self.hole_pos) < HOLE_R

    def get_params_vector(self):
        """Order: [sliding, torsional, rolling, timeconst, dampratio, mass]"""
        p = self.params
        return np.array(
            p["ground_friction"] + p["ground_solref"] + [p["ball_mass"]]
        )

    @staticmethod
    def params_from_vector(vec):
        return {
            "ground_friction": vec[0:3].tolist(),
            "ground_solref": vec[3:5].tolist(),
            "ball_mass": float(vec[5]),
        }
