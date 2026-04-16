import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "env" / "src"))

from env.src.golf import GolfEnv, DEFAULT_PARAMS, BALL_RADIUS
from env.src.course import (
    HOLE_R,
    MAX_BOUNCES,
    BALL_R,
    MIN_GAP,
    OBSTACLE_WALL_SPACING,
    OBSTACLE_POST_SPACING,
    _distribute_segment_budget,
    generate_course,
)


# Clearance thresholds scale with the course's (randomized) wall_t.
def _path_wall_clearance(spec):
    return BALL_R + MIN_GAP + spec.get("wall_t", 0.03) + 0.06


def _path_post_clearance(spec):
    return BALL_R + MIN_GAP + 0.06


def _corridor_obstacle_wall_clearance(spec):
    return 2 * spec.get("wall_t", 0.03) + OBSTACLE_WALL_SPACING


def _corridor_obstacle_post_clearance(spec):
    return spec.get("wall_t", 0.03) + OBSTACLE_POST_SPACING

SEED = 42
GEOM_TOL = 1e-6


def _corridor_hit(env, speed):
    """Hit along the first corridor segment so the ball doesn't bounce off walls."""
    target = env.waypoints[1] if len(env.waypoints) > 1 else env.hole_pos
    fwd = target - env.ball_start
    fwd = fwd / np.linalg.norm(fwd)
    return (fwd * speed).tolist()


def _wall_endpoints(wall):
    pos, angle_deg, hl = wall
    angle = np.radians(angle_deg)
    delta = hl * np.array([np.cos(angle), np.sin(angle)])
    return pos - delta, pos + delta


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _close(p, q, tol=GEOM_TOL):
    return np.linalg.norm(p - q) <= tol


def _on_segment(a, b, p, tol=GEOM_TOL):
    if abs(_cross2(b - a, p - a)) > tol:
        return False
    lo = np.minimum(a, b) - tol
    hi = np.maximum(a, b) + tol
    return np.all(p >= lo) and np.all(p <= hi)


def _bad_wall_overlap(w1, w2):
    p1, p2 = _wall_endpoints(w1)
    q1, q2 = _wall_endpoints(w2)

    d1 = p2 - p1
    d2 = q2 - q1

    if abs(_cross2(d1, d2)) <= GEOM_TOL and abs(_cross2(d1, q1 - p1)) <= GEOM_TOL:
        axis = 0 if abs(d1[0]) >= abs(d1[1]) else 1
        p_lo, p_hi = sorted((p1[axis], p2[axis]))
        q_lo, q_hi = sorted((q1[axis], q2[axis]))
        overlap = min(p_hi, q_hi) - max(p_lo, q_lo)
        return overlap > GEOM_TOL

    shared = (
        _close(p1, q1) or _close(p1, q2) or
        _close(p2, q1) or _close(p2, q2)
    )

    endpoints = ((p1, p2, q1, q2), (p2, p1, q1, q2), (q1, q2, p1, p2), (q2, q1, p1, p2))
    for point, other_end, seg_a, seg_b in endpoints:
        if _on_segment(seg_a, seg_b, point):
            if not (_close(point, seg_a) or _close(point, seg_b)):
                return True
            if not shared and not _close(point, other_end):
                return True

    o1 = _cross2(q1 - p1, d1)
    o2 = _cross2(q2 - p1, d1)
    o3 = _cross2(p1 - q1, d2)
    o4 = _cross2(p2 - q1, d2)
    return (o1 * o2 < -GEOM_TOL) and (o3 * o4 < -GEOM_TOL)


def _point_seg_dist(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def _seg_cross(a0, a1, b0, b1):
    d1 = a1 - a0
    d2 = b1 - b0
    denom = _cross2(d1, d2)
    if abs(denom) <= GEOM_TOL:
        return False
    delta = b0 - a0
    t = _cross2(delta, d2) / denom
    u = _cross2(delta, d1) / denom
    return 0 < t < 1 and 0 < u < 1


def _seg_dist(a0, a1, b0, b1):
    if _seg_cross(a0, a1, b0, b1):
        return 0.0
    return min(
        _point_seg_dist(a0, b0, b1),
        _point_seg_dist(a1, b0, b1),
        _point_seg_dist(b0, a0, a1),
        _point_seg_dist(b1, a0, a1),
    )


# --- physics sanity ---

def test_ball_stays_on_ground():
    env = GolfEnv(seed=SEED)
    traj = env.rollout(_corridor_hit(env, 3.0))
    z = traj["positions"][:, 2]
    assert np.all(z > 0)
    assert np.all(z < 0.15)


def test_ball_moves_and_stops():
    env = GolfEnv(seed=SEED)
    traj = env.rollout(_corridor_hit(env, 1.0))
    dist = np.linalg.norm(traj["positions"][-1][:2] - env.ball_start)
    assert dist > 0.1
    assert np.linalg.norm(traj["velocities"][-1]) < 0.01
    assert traj["times"][-1] < 10.0


def test_faster_hit_goes_farther():
    env = GolfEnv(seed=SEED)
    slow = env.rollout(_corridor_hit(env, 0.5))
    fast = env.rollout(_corridor_hit(env, 1.5))
    d_slow = np.linalg.norm(slow["positions"][-1][:2] - env.ball_start)
    d_fast = np.linalg.norm(fast["positions"][-1][:2] - env.ball_start)
    assert d_fast > d_slow


def test_friction_slick_vs_rough():
    hit = _corridor_hit(GolfEnv(seed=SEED), 1.0)
    slick = GolfEnv({"ground_friction": [0.3, 0.003, 0.005]}, seed=SEED)
    rough = GolfEnv({"ground_friction": [0.8, 0.008, 0.05]}, seed=SEED)
    ds = np.linalg.norm(slick.rollout(hit)["positions"][-1][:2] - slick.ball_start)
    dr = np.linalg.norm(rough.rollout(hit)["positions"][-1][:2] - rough.ball_start)
    assert ds > dr


def test_solref_affects_trajectory():
    hit = GolfEnv(seed=SEED, n_bounces=2).solve()
    stiff = GolfEnv({"ground_solref": [0.005, 1.0]}, seed=SEED, n_bounces=2)
    soft = GolfEnv({"ground_solref": [0.05, 1.0]}, seed=SEED, n_bounces=2)
    t1 = stiff.rollout(hit)
    t2 = soft.rollout(hit)
    n = min(len(t1["times"]), len(t2["times"]))
    diff = np.max(np.abs(t1["positions"][:n] - t2["positions"][:n]))
    assert diff > 1e-6


# --- determinism ---

def test_deterministic():
    env = GolfEnv(seed=SEED)
    hit = _corridor_hit(env, 2.0)
    t1 = env.rollout(hit)
    t2 = env.rollout(hit)
    assert np.allclose(t1["positions"], t2["positions"])
    assert np.allclose(t1["velocities"], t2["velocities"])


def test_same_seed_same_course():
    e1 = GolfEnv(seed=SEED)
    e2 = GolfEnv(seed=SEED)
    assert np.allclose(e1.ball_start, e2.ball_start)
    assert np.allclose(e1.hole_pos, e2.hole_pos)
    hit = _corridor_hit(e1, 2.0)
    assert np.allclose(e1.rollout(hit)["positions"], e2.rollout(hit)["positions"])


def test_different_seeds_different_courses():
    e1 = GolfEnv(seed=0)
    e2 = GolfEnv(seed=50)
    assert not np.allclose(e1.hole_pos, e2.hole_pos)


# --- trajectory shape ---

def test_trajectory_keys_and_shapes():
    env = GolfEnv(seed=SEED)
    traj = env.rollout(_corridor_hit(env, 2.0))
    assert set(traj.keys()) == {"times", "positions", "velocities"}
    n = len(traj["times"])
    assert traj["positions"].shape == (n, 3)
    assert traj["velocities"].shape == (n, 3)
    assert np.all(np.diff(traj["times"]) > 0)


def test_dt_record_spacing():
    env = GolfEnv(seed=SEED)
    traj = env.rollout(_corridor_hit(env, 1.0), dt_record=0.05)
    diffs = np.diff(traj["times"][:-1])
    assert np.allclose(diffs, 0.05, atol=0.005)


def test_initial_position_is_ball_start():
    env = GolfEnv(seed=SEED)
    traj = env.rollout(_corridor_hit(env, 1.0))
    start = traj["positions"][0]
    assert np.allclose(start[:2], env.ball_start, atol=1e-4)
    assert abs(start[2] - BALL_RADIUS) < 1e-4


# --- course generation ---

def test_hole_far_from_start():
    for s in range(10):
        env = GolfEnv(seed=s)
        d = np.linalg.norm(env.hole_pos - env.ball_start)
        assert d >= 0.5, f"seed {s}: hole too close ({d:.2f}m)"


def test_hole_inside_arena():
    for s in range(10):
        env = GolfEnv(seed=s)
        assert np.all(np.abs(env.hole_pos) < 2.5)


def test_ball_in_hole_check():
    env = GolfEnv(seed=SEED)
    assert env.ball_in_hole(env.hole_pos)
    assert not env.ball_in_hole(env.ball_start)


def test_multiple_seeds_build():
    for s in range(20):
        env = GolfEnv(seed=s)
        assert env.model is not None


def test_all_bounce_counts_build():
    for n_bounces in range(MAX_BOUNCES + 1):
        spec = generate_course(seed=100 + n_bounces, n_bounces=n_bounces)
        assert spec["n_bounces"] == n_bounces
        assert len(spec["waypoints"]) == n_bounces + 2


def test_new_course_changes_layout():
    env = GolfEnv(seed=0)
    hole1 = env.hole_pos.copy()
    env.new_course(seed=99)
    assert not np.allclose(hole1, env.hole_pos)


def test_walls_only_touch_at_endpoints():
    for seed in range(50):
        walls = generate_course(seed)["walls"]
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                assert not _bad_wall_overlap(walls[i], walls[j]), (
                    f"seed {seed}: walls {i} and {j} overlap"
                )


def test_obstacles_stay_clear_of_solution_path():
    for seed in range(40):
        spec = generate_course(seed)
        path_wall = _path_wall_clearance(spec)
        path_post = _path_post_clearance(spec)
        path_nodes = [spec["start"]] + spec["waypoints"][1:-1] + [spec["hole"]]
        path_segments = list(zip(path_nodes[:-1], path_nodes[1:]))

        for wall in spec["obstacle_walls"]:
            w0, w1 = _wall_endpoints(wall)
            for p0, p1 in path_segments:
                assert _seg_dist(w0, w1, p0, p1) >= path_wall - 1e-6, (
                    f"seed {seed}: obstacle wall too close to solution path"
                )

        for center, radius in spec["obstacle_posts"]:
            for p0, p1 in path_segments:
                assert _point_seg_dist(center, p0, p1) >= radius + path_post - 1e-6, (
                    f"seed {seed}: obstacle post blocks solution path"
                )


def test_obstacle_density_stays_bounded():
    for seed in range(40):
        spec = generate_course(seed)
        assert len(spec["obstacle_patterns"]) <= 6, (
            f"seed {seed}: too many obstacle patterns ({len(spec['obstacle_patterns'])})"
        )


def test_obstacle_budget_spreads_across_segments():
    rng = np.random.default_rng(0)
    counts = _distribute_segment_budget(rng, [1.0, 1.1, 0.9, 1.2], 3)
    assert sum(counts) == 3
    assert max(counts) <= 1

    rng = np.random.default_rng(1)
    counts = _distribute_segment_budget(rng, [2.0, 2.0, 2.0], 5)
    assert sum(counts) == 5
    assert max(counts) - min(counts) <= 1


def test_obstacles_stay_clear_of_corridor_walls():
    for seed in range(40):
        spec = generate_course(seed)
        co_wall = _corridor_obstacle_wall_clearance(spec)
        co_post = _corridor_obstacle_post_clearance(spec)
        corridor_count = len(spec["walls"]) - len(spec["obstacle_walls"])
        corridor_walls = spec["walls"][:corridor_count]

        for obstacle_wall in spec["obstacle_walls"]:
            w0, w1 = _wall_endpoints(obstacle_wall)
            for corridor_wall in corridor_walls:
                c0, c1 = _wall_endpoints(corridor_wall)
                assert _seg_dist(w0, w1, c0, c1) >= co_wall - 1e-6, (
                    f"seed {seed}: obstacle wall overlaps corridor wall"
                )

        for center, radius in spec["obstacle_posts"]:
            for corridor_wall in corridor_walls:
                c0, c1 = _wall_endpoints(corridor_wall)
                assert _point_seg_dist(center, c0, c1) >= radius + co_post - 1e-6, (
                    f"seed {seed}: obstacle post overlaps corridor wall"
                )


def test_solve_hits_hole_across_many_seeds():
    for seed in range(25):
        env = GolfEnv(seed=seed)
        traj = env.rollout(env.solve(), max_time=20.0)
        final = traj["positions"][-1][:2]
        assert np.linalg.norm(final - env.hole_pos) < HOLE_R, (
            f"seed {seed}: solution missed hole by "
            f"{np.linalg.norm(final - env.hole_pos):.4f}m"
        )


# --- params vector ---

def test_params_roundtrip():
    env = GolfEnv(seed=SEED)
    vec = env.get_params_vector()
    rebuilt = GolfEnv.params_from_vector(vec)
    for key in DEFAULT_PARAMS:
        assert np.allclose(rebuilt[key], DEFAULT_PARAMS[key])


def test_params_vector_length():
    assert len(GolfEnv(seed=SEED).get_params_vector()) == 6


def test_custom_params_applied():
    custom = {"ground_friction": [0.9, 0.02, 0.03], "ball_mass": 0.1}
    env = GolfEnv(custom, seed=SEED)
    assert env.params["ground_friction"] == [0.9, 0.02, 0.03]
    assert env.params["ball_mass"] == 0.1
    assert env.params["ground_solref"] == DEFAULT_PARAMS["ground_solref"]


# --- edge cases ---

def test_zero_velocity():
    env = GolfEnv(seed=SEED)
    traj = env.rollout([0.0, 0.0])
    dist = np.linalg.norm(traj["positions"][-1][:2] - env.ball_start)
    assert dist < 0.01


def test_short_max_time():
    env = GolfEnv(seed=SEED)
    traj = env.rollout(_corridor_hit(env, 5.0), max_time=0.1)
    assert traj["times"][-1] <= 0.11
    assert np.linalg.norm(traj["velocities"][-1]) > 0.1


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
        print(f"[OK] {t.__name__}")
    print(f"\n{len(ALL_TESTS)} tests passed")
