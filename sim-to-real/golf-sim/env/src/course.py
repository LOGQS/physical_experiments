import numpy as np

ARENA = 2.3
CORRIDOR_W = 0.35
WALL_H = 0.05
WALL_T = 0.03
HOLE_R = 0.054
BALL_R = 0.02135
MIN_CLEARANCE = 0.15
MIN_GAP = BALL_R * 3
BOUNCE_RESTITUTION = 0.98
MAX_BOUNCES = 6
OBSTACLE_CLEARANCE = BALL_R + MIN_GAP + 0.02
OBSTACLE_BOUNDARY_MARGIN = 0.04
OBSTACLE_EDGE_MARGIN = 0.22
OBSTACLE_POST_SPACING = 0.03
OBSTACLE_WALL_SPACING = 0.03
MIN_WALL_LENGTH = 1e-4
PATH_WALL_CLEARANCE = BALL_R + MIN_GAP + WALL_T + 0.06
PATH_POST_CLEARANCE = BALL_R + MIN_GAP + 0.06
CORRIDOR_OBSTACLE_WALL_CLEARANCE = 2 * WALL_T + OBSTACLE_WALL_SPACING
CORRIDOR_OBSTACLE_POST_CLEARANCE = WALL_T + OBSTACLE_POST_SPACING
COURSE_OBSTACLE_BUDGET_BASE = 2
COURSE_OBSTACLE_BUDGET_PER_M = 1.1
SEGMENT_PATTERN_MAX = 2


_RANDOMIZED_GLOBALS = (
    "CORRIDOR_W", "WALL_H", "WALL_T", "MIN_CLEARANCE",
    "PATH_WALL_CLEARANCE", "CORRIDOR_OBSTACLE_WALL_CLEARANCE",
    "CORRIDOR_OBSTACLE_POST_CLEARANCE",
    "COURSE_OBSTACLE_BUDGET_BASE", "COURSE_OBSTACLE_BUDGET_PER_M",
    "SEGMENT_PATTERN_MAX", "OBSTACLE_EDGE_MARGIN",
)


def _sample_cfg(rng):
    """Sample per-course scene parameters. Ranges chosen so all archetypes
    still satisfy miter-reach and arena-fit constraints."""
    return {
        "corridor_w":     float(rng.uniform(0.25, 0.38)),
        "wall_h":         float(rng.uniform(0.035, 0.11)),
        "wall_t":         float(rng.uniform(0.02, 0.045)),
        "min_clearance":  float(rng.uniform(0.10, 0.25)),
        "obstacle_base":  float(rng.uniform(1.0, 3.2)),
        "obstacle_per_m": float(rng.uniform(0.5, 1.6)),
        "segment_pattern_max": int(rng.choice([1, 2, 3])),
        "obstacle_edge_margin": float(rng.uniform(0.15, 0.30)),
    }


def generate_course(seed=None, n_bounces=None):
    global CORRIDOR_W, WALL_H, WALL_T, MIN_CLEARANCE
    global PATH_WALL_CLEARANCE, CORRIDOR_OBSTACLE_WALL_CLEARANCE
    global CORRIDOR_OBSTACLE_POST_CLEARANCE
    global COURSE_OBSTACLE_BUDGET_BASE, COURSE_OBSTACLE_BUDGET_PER_M
    global SEGMENT_PATTERN_MAX, OBSTACLE_EDGE_MARGIN

    saved = {name: globals()[name] for name in _RANDOMIZED_GLOBALS}

    rng = np.random.default_rng(seed)
    if n_bounces is None:
        target_bounces = int(rng.integers(0, MAX_BOUNCES + 1))
    else:
        target_bounces = int(n_bounces)
        if not 0 <= target_bounces <= MAX_BOUNCES:
            raise ValueError(f"n_bounces must be in [0, {MAX_BOUNCES}]")

    cfg = _sample_cfg(rng)
    CORRIDOR_W = cfg["corridor_w"]
    WALL_H = cfg["wall_h"]
    WALL_T = cfg["wall_t"]
    MIN_CLEARANCE = cfg["min_clearance"]
    COURSE_OBSTACLE_BUDGET_BASE = cfg["obstacle_base"]
    COURSE_OBSTACLE_BUDGET_PER_M = cfg["obstacle_per_m"]
    SEGMENT_PATTERN_MAX = cfg["segment_pattern_max"]
    OBSTACLE_EDGE_MARGIN = cfg["obstacle_edge_margin"]
    PATH_WALL_CLEARANCE = BALL_R + MIN_GAP + WALL_T + 0.06
    CORRIDOR_OBSTACLE_WALL_CLEARANCE = 2 * WALL_T + OBSTACLE_WALL_SPACING
    CORRIDOR_OBSTACLE_POST_CLEARANCE = WALL_T + OBSTACLE_POST_SPACING

    try:
        archetype = None
        for _ in range(1000):
            result = _make_bounce_path(rng, target_bounces)
            if result is None:
                continue
            waypoints, archetype = result
            if not _path_valid(waypoints):
                continue
            corridor_walls = _corridor(waypoints)
            if not _walls_valid(corridor_walls):
                continue
            collision_walls = _corridor_collision_walls(waypoints)
            break
        else:
            raise RuntimeError(f"failed to generate a {target_bounces}-bounce course")

        fwd = waypoints[1] - waypoints[0]
        fwd /= np.linalg.norm(fwd)
        start = waypoints[0] + MIN_CLEARANCE * fwd

        bwd = waypoints[-2] - waypoints[-1]
        bwd /= np.linalg.norm(bwd)
        hole = waypoints[-1] + MIN_CLEARANCE * bwd

        bump_walls, posts, obstacle_patterns = _obstacles(
            rng, waypoints, start, hole, corridor_walls
        )

        solution_vel = _compute_solution_velocity(waypoints, start, hole)

        return {
            "start": start, "hole": hole,
            "walls": corridor_walls + bump_walls,
            "collision_walls": collision_walls,
            "posts": posts,
            "obstacle_walls": bump_walls,
            "obstacle_posts": posts,
            "obstacle_patterns": obstacle_patterns,
            "waypoints": waypoints,
            "n_bounces": target_bounces,
            "archetype": archetype,
            "solution_velocity": solution_vel,
            "corridor_w": CORRIDOR_W,
            "wall_h": WALL_H,
            "wall_t": WALL_T,
            "min_clearance": MIN_CLEARANCE,
        }
    finally:
        for name, val in saved.items():
            globals()[name] = val


# --- path: archetype library ---
#
# Each archetype builds a small set of local-coordinate waypoints with its own
# shape rules (lengths, turn signs, turn magnitudes). `_finalize` then rotates
# the local path by a random heading and shifts it into the arena.
# The global `_angles_ok` and `_segments_ok` are loose safety nets — each
# archetype is responsible for staying within its intended geometry.

def _make_bounce_path(rng, n_bounces):
    name, local_wps = _pick_archetype(rng, n_bounces)
    if local_wps is None:
        return None
    waypoints = _finalize(rng, local_wps)
    if waypoints is None:
        return None
    if not _segments_ok(waypoints):
        return None
    if not _angles_ok(waypoints):
        return None
    return waypoints, name


def _finalize(rng, local_wps):
    bound = ARENA - CORRIDOR_W - MIN_CLEARANCE
    local_arr = np.array(local_wps)
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    rotated = (R @ local_arr.T).T
    mins = rotated.min(axis=0)
    maxs = rotated.max(axis=0)
    if np.any(maxs - mins > 2 * bound):
        return None

    shift_lo = -bound - mins
    shift_hi = bound - maxs
    shift = rng.uniform(shift_lo, shift_hi)
    pts = rotated + shift
    return [np.array(p) for p in pts]


def _path_from_headings(segment_lengths, turn_angles):
    heading = 0.0
    pos = np.array([0.0, 0.0])
    pts = [pos.copy()]
    for i, length in enumerate(segment_lengths):
        d = np.array([np.cos(heading), np.sin(heading)])
        pos = pos + length * d
        pts.append(pos.copy())
        if i < len(turn_angles):
            heading += turn_angles[i]
    return pts


def _straight(rng):
    return [np.array([0.0, 0.0]), np.array([rng.uniform(2.2, 3.3), 0.0])]


def _two_seg(rng, turn_deg, total_range, ratio_range=(0.22, 0.78)):
    # Total path length + U-shaped split → bounce lands anywhere between 1/4
    # and 3/4 of the way along start→hole, not locked near the midpoint.
    total = rng.uniform(*total_range)
    ratio = float(np.clip(rng.beta(0.7, 0.7), *ratio_range))
    d1 = total * ratio
    d2 = total * (1 - ratio)
    side = rng.choice([-1, 1])
    return _path_from_headings([d1, d2], [side * np.radians(turn_deg)])


def _three_seg(rng, a1_deg, a2_deg, t1_sign, t2_sign, total_range):
    # Each segment reserves a miter-reach-safe minimum; the remainder is
    # distributed by Dirichlet(0.8), which concentrates mass at one or two
    # segments so bounces cluster/spread instead of sitting at 1/3, 2/3.
    r1 = CORRIDOR_W * np.tan(np.radians(a1_deg) / 2)
    r2 = CORRIDOR_W * np.tan(np.radians(a2_deg) / 2)
    mins = [
        max(0.35, r1 * 1.15),
        max(0.35, (r1 + r2) * 1.15),
        max(0.35, r2 * 1.15),
    ]
    reserved = sum(mins)
    lo = max(total_range[0], reserved + 0.3)
    hi = total_range[1]
    if lo > hi:
        return None
    total = rng.uniform(lo, hi)
    extra = total - reserved
    ratios = rng.dirichlet([0.8, 0.8, 0.8])
    segs = [m + r * extra for m, r in zip(mins, ratios)]
    return _path_from_headings(segs, [t1_sign * np.radians(a1_deg),
                                      t2_sign * np.radians(a2_deg)])


def _dogleg(rng):
    # Beta(2.5, 2.5) is bell-shaped around 0.5 → turn concentrates near 90°,
    # the "clean L" most mini-golf doglegs use. Still reaches 70° and 115°.
    turn_deg = 70 + 45 * float(rng.beta(2.5, 2.5))
    return _two_seg(rng, turn_deg, (2.2, 3.6))


def _hairpin(rng):
    # Kept as direct sampling because we need d2 < d1 (return leg shorter)
    # and a specific start-hole separation. Still widened: d2/d1 ratio varies
    # 0.2 → 0.55 instead of the old near-fixed ~0.5.
    d1 = rng.uniform(1.8, 2.6)
    d2 = rng.uniform(0.4, d1 * 0.55)
    side = rng.choice([-1, 1])
    turn = side * np.radians(rng.uniform(140, 165))
    return _path_from_headings([d1, d2], [turn])


def _bank(rng):
    return _two_seg(rng, rng.uniform(30, 60), (2.6, 3.4))


def _s_curve(rng):
    side = rng.choice([-1, 1])
    return _three_seg(rng, rng.uniform(60, 100), rng.uniform(60, 100),
                      side, -side, (2.4, 4.0))


def _cardinal(rng):
    side = rng.choice([-1, 1])
    return _three_seg(rng, rng.uniform(45, 75), rng.uniform(45, 75),
                      side, side, (2.2, 3.4))


def _pocket(rng):
    side = rng.choice([-1, 1])
    return _three_seg(rng, rng.uniform(50, 95), rng.uniform(95, 135),
                      side, -side, (2.2, 3.6))


def _zigzag(rng, n_bounces):
    # Ranges tuned empirically so miter corners don't overlap
    # (need seg > 2 * CORRIDOR_W * tan(turn/2)) and the folded path still fits
    # the arena. Bigger bounce counts need shorter segments and sharper turns.
    if n_bounces <= 3:
        seg_min, seg_cap = 0.88, 1.25
        turn_min, turn_max = 70, 115
    elif n_bounces == 4:
        seg_min, seg_cap = 0.7, 1.0
        turn_min, turn_max = 70, 95
    elif n_bounces == 5:
        seg_min, seg_cap = 0.7, 1.0
        turn_min, turn_max = 85, 115
    else:
        seg_min, seg_cap = 0.6, 0.85
        turn_min, turn_max = 85, 100

    segs = list(rng.uniform(seg_min, seg_cap, size=n_bounces + 1))
    side = rng.choice([-1, 1])
    turns = []
    for _ in range(n_bounces):
        turns.append(side * np.radians(rng.uniform(turn_min, turn_max)))
        side = -side
    return _path_from_headings(segs, turns)


def _spiral(rng, n_bounces):
    # All turns same side → consistent curl (n=3 only: 4+ same-side bounces
    # always have non-adjacent outer walls cross in this arena/corridor).
    # Two modes: uniform segments (even curl) and decreasing segments that
    # curl inward like a nautilus.
    if n_bounces != 3:
        return None
    side = rng.choice([-1, 1])
    total_rot_deg = float(rng.uniform(150, 210))
    base_deg = total_rot_deg / n_bounces
    turns = [side * np.radians(base_deg + float(rng.uniform(-6, 6)))
             for _ in range(n_bounces)]
    seg_base = float(rng.uniform(1.2, 1.5))
    if rng.random() < 0.5:
        segs = [seg_base * (1.25 - 0.15 * i) for i in range(n_bounces + 1)]
        segs = [s * float(rng.uniform(0.9, 1.1)) for s in segs]
    else:
        segs = list(rng.uniform(seg_base * 0.85, seg_base * 1.1, size=n_bounces + 1))
    return _path_from_headings(segs, turns)


_ARCHETYPES = {
    0: [(_straight, 1.0)],
    1: [(_dogleg, 3.0), (_hairpin, 1.0), (_bank, 2.0)],
    2: [(_s_curve, 2.0), (_cardinal, 1.5), (_pocket, 1.5)],
}


def _pick_archetype(rng, n_bounces):
    if n_bounces in _ARCHETYPES:
        choices = _ARCHETYPES[n_bounces]
        fns = [c[0] for c in choices]
        weights = np.array([c[1] for c in choices], dtype=float)
        weights /= weights.sum()
        idx = int(rng.choice(len(fns), p=weights))
        fn = fns[idx]
        return fn.__name__.lstrip("_"), fn(rng)
    # 3+ bounces: zigzag always; spiral also available at n=3 (same-side curl
    # only fits the arena without wall overlap at 3 bounces).
    if n_bounces == 3 and rng.random() < 0.45:
        return "spiral", _spiral(rng, n_bounces)
    return "zigzag", _zigzag(rng, n_bounces)


def _segments_ok(pts):
    for i in range(len(pts) - 1):
        d = np.linalg.norm(pts[i + 1] - pts[i])
        if d < 0.3 or d > 3.5:
            return False
    return True


def _angles_ok(pts):
    # Loose guardrail: rejects near-parallel (<~23°) and near-reversed (>~168°)
    # turns that would produce degenerate corridor geometry. Individual
    # archetypes stay well inside these bounds.
    for i in range(1, len(pts) - 1):
        d_in = pts[i] - pts[i - 1]
        d_out = pts[i + 1] - pts[i]
        cos_a = np.dot(d_in, d_out) / (np.linalg.norm(d_in) * np.linalg.norm(d_out))
        if cos_a > 0.92 or cos_a < -0.98:
            return False
    return True


def _path_valid(waypoints):
    n = len(waypoints) - 1
    for i in range(n):
        for j in range(i + 2, n):
            if _seg_cross(waypoints[i], waypoints[i + 1],
                          waypoints[j], waypoints[j + 1]):
                return False
    return True


def _seg_cross(p1, p2, p3, p4):
    d1, d2 = p2 - p1, p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-10:
        return False
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    return 0 < t < 1 and 0 < u < 1


def _point_seg_dist(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def _seg_dist(a0, a1, b0, b1):
    if _seg_cross(a0, a1, b0, b1):
        return 0.0
    return min(
        _point_seg_dist(a0, b0, b1),
        _point_seg_dist(a1, b0, b1),
        _point_seg_dist(b0, a0, a1),
        _point_seg_dist(b1, a0, a1),
    )


def _close_pts(p, q, tol=1e-6):
    return np.linalg.norm(p - q) <= tol


def _point_on_segment(a, b, p, tol=1e-6):
    if abs(_cross2(b - a, p - a)) > tol:
        return False
    lo = np.minimum(a, b) - tol
    hi = np.maximum(a, b) + tol
    return np.all(p >= lo) and np.all(p <= hi)


def _bad_wall_overlap(w1, w2, tol=1e-6):
    p1, p2 = _wall_endpoints(w1)
    q1, q2 = _wall_endpoints(w2)

    d1 = p2 - p1
    d2 = q2 - q1

    if abs(_cross2(d1, d2)) <= tol and abs(_cross2(d1, q1 - p1)) <= tol:
        axis = 0 if abs(d1[0]) >= abs(d1[1]) else 1
        p_lo, p_hi = sorted((p1[axis], p2[axis]))
        q_lo, q_hi = sorted((q1[axis], q2[axis]))
        overlap = min(p_hi, q_hi) - max(p_lo, q_lo)
        return overlap > tol

    shared = (
        _close_pts(p1, q1, tol) or _close_pts(p1, q2, tol) or
        _close_pts(p2, q1, tol) or _close_pts(p2, q2, tol)
    )

    endpoints = (
        (p1, p2, q1, q2),
        (p2, p1, q1, q2),
        (q1, q2, p1, p2),
        (q2, q1, p1, p2),
    )
    for point, other_end, seg_a, seg_b in endpoints:
        if _point_on_segment(seg_a, seg_b, point, tol):
            if not (_close_pts(point, seg_a, tol) or _close_pts(point, seg_b, tol)):
                return True
            if not shared and not _close_pts(point, other_end, tol):
                return True

    o1 = _cross2(q1 - p1, d1)
    o2 = _cross2(q2 - p1, d1)
    o3 = _cross2(p1 - q1, d2)
    o4 = _cross2(p2 - q1, d2)
    return (o1 * o2 < -tol) and (o3 * o4 < -tol)


def _walls_valid(walls):
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            if _bad_wall_overlap(walls[i], walls[j]):
                return False
    return True


# --- corridor ---

def _seg_geo(a, b):
    fwd = b - a
    length = np.linalg.norm(fwd)
    fwd_hat = fwd / length
    perp = np.array([-fwd_hat[1], fwd_hat[0]])
    return fwd_hat, perp, length


def _wall_from_pts(p1, p2):
    mid = (p1 + p2) / 2
    d = p2 - p1
    length = np.linalg.norm(d)
    if length <= MIN_WALL_LENGTH:
        return None
    return (mid, np.degrees(np.arctan2(d[1], d[0])), length / 2)


def _wall_endpoints(wall):
    pos, angle_deg, hl = wall
    angle = np.radians(angle_deg)
    delta = hl * np.array([np.cos(angle), np.sin(angle)])
    return pos - delta, pos + delta


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _line_intersection(p1, d1, p2, d2):
    denom = _cross2(d1, d2)
    if abs(denom) < 1e-10:
        return (p1 + p2) / 2
    t = _cross2(p2 - p1, d2) / denom
    return p1 + t * d1


def _bounce_line(p_prev, p_bounce, p_next, contact=False):
    d_in = p_bounce - p_prev
    d_in /= np.linalg.norm(d_in)
    d_out = p_next - p_bounce
    d_out /= np.linalg.norm(d_out)

    # For ideal reflection the wall normal bisects the velocity change.
    normal = d_out - d_in
    normal /= np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    point = p_bounce if contact else p_bounce - (BALL_R + WALL_T) * normal
    return point, tangent


def _local_to_world(origin, fwd, perp, p):
    return origin + p[0] * fwd + p[1] * perp


def _world_wall(origin, fwd, perp, p1, p2):
    return _wall_from_pts(
        _local_to_world(origin, fwd, perp, p1),
        _local_to_world(origin, fwd, perp, p2),
    )


def _world_post(origin, fwd, perp, center, radius):
    return (_local_to_world(origin, fwd, perp, center), radius)


def _segment_slots(rng, seg_len, count, jitter=0.06):
    xs = np.linspace(OBSTACLE_EDGE_MARGIN, seg_len - OBSTACLE_EDGE_MARGIN, count)
    if count > 1:
        span = min(jitter, 0.25 * seg_len / count)
        xs += rng.uniform(-span, span, size=count)
        xs = np.clip(xs, OBSTACLE_EDGE_MARGIN, seg_len - OBSTACLE_EDGE_MARGIN)
        xs.sort()
    return xs


def _pattern_slalom_posts(rng, seg_len):
    count = int(rng.integers(2, min(4, max(3, int(seg_len / 0.4) + 1))))
    xs = _segment_slots(rng, seg_len, count)
    first_side = rng.choice([-1, 1])
    posts = []
    for idx, x in enumerate(xs):
        radius = rng.uniform(0.028, 0.048)
        min_y = OBSTACLE_CLEARANCE + radius
        max_y = CORRIDOR_W - OBSTACLE_BOUNDARY_MARGIN - radius
        if max_y <= min_y:
            return None
        y = first_side * ((-1) ** idx) * rng.uniform(min_y, max_y)
        posts.append((np.array([x, y]), radius))
    return {"name": "slalom_posts", "walls": [], "posts": posts}


def _pattern_gate_posts(rng, seg_len):
    gate_count = int(rng.integers(1, 3 if seg_len > 1.0 else 2))
    xs = _segment_slots(rng, seg_len, gate_count, jitter=0.08)
    posts = []
    for x in xs:
        radius = rng.uniform(0.026, 0.045)
        gap_half = OBSTACLE_CLEARANCE + rng.uniform(0.01, 0.05)
        y = gap_half + radius
        if y + radius > CORRIDOR_W - OBSTACLE_BOUNDARY_MARGIN:
            continue
        posts.append((np.array([x, y]), radius))
        posts.append((np.array([x, -y]), radius))
    if not posts:
        return None
    return {"name": "gate_posts", "walls": [], "posts": posts}


def _pattern_side_cluster(rng, seg_len):
    side = rng.choice([-1, 1])
    count = int(rng.integers(2, 4))
    xs = _segment_slots(rng, seg_len, count, jitter=0.09)

    posts = []
    for x in xs:
        radius = rng.uniform(0.026, 0.044)
        min_y = OBSTACLE_CLEARANCE + radius + 0.02
        max_y = CORRIDOR_W - OBSTACLE_BOUNDARY_MARGIN - radius
        if max_y <= min_y:
            return None
        y = side * rng.uniform(min_y, max_y)
        posts.append((np.array([x, y]), radius))
    return {"name": "side_cluster", "walls": [], "posts": posts}


def _distribute_segment_budget(rng, segment_lengths, global_budget):
    target = [0] * len(segment_lengths)
    eligible = [
        idx for idx, seg_len in enumerate(segment_lengths)
        if seg_len >= 0.45
    ]
    if not eligible or global_budget <= 0:
        return target

    capacities = {
        idx: min(SEGMENT_PATTERN_MAX, max(1, int(np.round(segment_lengths[idx] / 0.95))))
        for idx in eligible
    }
    remaining = min(global_budget, sum(capacities.values()))
    order = list(rng.permutation(eligible))

    # Spread the first pass across as many segments as possible before giving
    # any segment a second pattern.
    while remaining > 0:
        made_progress = False
        for idx in order:
            if target[idx] >= capacities[idx]:
                continue
            target[idx] += 1
            remaining -= 1
            made_progress = True
            if remaining <= 0:
                break
        if not made_progress:
            break

    return target


def _pattern_priority(rng, builders, usage_counts, local_names):
    shuffled = list(rng.permutation(builders))
    return sorted(
        shuffled,
        key=lambda builder: (
            usage_counts.get(builder.__name__, 0),
            builder.__name__ in local_names,
        ),
    )


def _pattern_staggered_walls(rng, seg_len):
    max_count = 3 if seg_len > 1.0 else 2
    count = int(rng.integers(2, max_count + 1))
    xs = _segment_slots(rng, seg_len, count)
    first_side = rng.choice([-1, 1])
    walls = []
    for idx, x in enumerate(xs):
        side = first_side * ((-1) ** idx)
        length = rng.uniform(0.12, min(0.28, 0.35 * seg_len))
        angle = np.radians(rng.uniform(-40, 40))
        center_y_mag = rng.uniform(
            OBSTACLE_CLEARANCE + 0.05,
            CORRIDOR_W - OBSTACLE_BOUNDARY_MARGIN - 0.05,
        )
        center = np.array([x, side * center_y_mag])
        tangent = np.array([np.cos(angle), np.sin(angle)])
        if side * tangent[1] < 0:
            tangent[1] *= -1
        p1 = center - 0.5 * length * tangent
        p2 = center + 0.5 * length * tangent
        walls.append((p1, p2))
    return {"name": "staggered_walls", "walls": walls, "posts": []}


def _pattern_pinball_pair(rng, seg_len):
    x0 = rng.uniform(0.24 * seg_len, 0.76 * seg_len)
    dx = rng.uniform(0.12, 0.22)
    side = rng.choice([-1, 1])

    radius = rng.uniform(0.028, 0.046)
    post_min_y = OBSTACLE_CLEARANCE + radius + 0.01
    post_max_y = CORRIDOR_W - OBSTACLE_BOUNDARY_MARGIN - radius
    if post_max_y <= post_min_y:
        return None
    post = (np.array([x0 - dx / 2, side * rng.uniform(post_min_y, post_max_y)]), radius)

    length = rng.uniform(0.12, min(0.24, 0.3 * seg_len))
    other_side = -side
    wall_center_y_mag = rng.uniform(
        OBSTACLE_CLEARANCE + 0.04,
        CORRIDOR_W - OBSTACLE_BOUNDARY_MARGIN - 0.05,
    )
    center = np.array([x0 + dx / 2, other_side * wall_center_y_mag])
    angle = np.radians(rng.uniform(-35, 35))
    tangent = np.array([np.cos(angle), np.sin(angle)])
    if other_side * tangent[1] < 0:
        tangent[1] *= -1
    wall = (center - 0.5 * length * tangent, center + 0.5 * length * tangent)
    return {"name": "pinball_pair", "walls": [wall], "posts": [post]}


def _pattern_midlane_boxout(rng, seg_len):
    x = rng.uniform(0.28 * seg_len, 0.72 * seg_len)
    width = rng.uniform(0.14, 0.24)
    gap_half = OBSTACLE_CLEARANCE + rng.uniform(0.015, 0.045)
    wall_len = rng.uniform(0.12, min(0.22, 0.28 * seg_len))
    walls = []
    for side in (-1, 1):
        y = side * (gap_half + wall_len / 2)
        if abs(y) + wall_len / 2 > CORRIDOR_W - OBSTACLE_BOUNDARY_MARGIN:
            return None
        p1 = np.array([x - width / 2, y - wall_len / 2])
        p2 = np.array([x + width / 2, y + wall_len / 2])
        walls.append((p1, p2))
    return {"name": "midlane_boxout", "walls": walls, "posts": []}


def _candidate_valid(candidate, local_walls, local_posts, seg_len, origin, fwd, perp,
                     start, hole, path_segments, corridor_segments):
    for p1, p2 in candidate["walls"]:
        if (min(p1[0], p2[0]) < OBSTACLE_EDGE_MARGIN or
                max(p1[0], p2[0]) > seg_len - OBSTACLE_EDGE_MARGIN):
            return False
        if max(abs(p1[1]), abs(p2[1])) > CORRIDOR_W - OBSTACLE_BOUNDARY_MARGIN:
            return False
        if min(abs(p1[1]), abs(p2[1])) < OBSTACLE_CLEARANCE:
            return False

        wp1 = _local_to_world(origin, fwd, perp, p1)
        wp2 = _local_to_world(origin, fwd, perp, p2)
        if (_point_seg_dist(start, wp1, wp2) < MIN_CLEARANCE * 1.8 or
                _point_seg_dist(hole, wp1, wp2) < MIN_CLEARANCE * 1.8):
            return False
        for path_a, path_b in path_segments:
            if _seg_dist(wp1, wp2, path_a, path_b) < PATH_WALL_CLEARANCE:
                return False
        for corr_a, corr_b in corridor_segments:
            if _seg_dist(wp1, wp2, corr_a, corr_b) < CORRIDOR_OBSTACLE_WALL_CLEARANCE:
                return False

        for q1, q2 in local_walls:
            if _seg_dist(p1, p2, q1, q2) < OBSTACLE_WALL_SPACING:
                return False
        for c, r in local_posts:
            if _point_seg_dist(c, p1, p2) < r + OBSTACLE_POST_SPACING:
                return False

    for center, radius in candidate["posts"]:
        if center[0] < OBSTACLE_EDGE_MARGIN or center[0] > seg_len - OBSTACLE_EDGE_MARGIN:
            return False
        if abs(center[1]) - radius < OBSTACLE_CLEARANCE:
            return False
        if abs(center[1]) + radius > CORRIDOR_W - OBSTACLE_BOUNDARY_MARGIN:
            return False

        wcenter = _local_to_world(origin, fwd, perp, center)
        if (np.linalg.norm(wcenter - start) < MIN_CLEARANCE * 1.8 or
                np.linalg.norm(wcenter - hole) < MIN_CLEARANCE * 1.8):
            return False
        for path_a, path_b in path_segments:
            if _point_seg_dist(wcenter, path_a, path_b) < radius + PATH_POST_CLEARANCE:
                return False
        for corr_a, corr_b in corridor_segments:
            if _point_seg_dist(wcenter, corr_a, corr_b) < radius + CORRIDOR_OBSTACLE_POST_CLEARANCE:
                return False

        for c, r in local_posts:
            if np.linalg.norm(center - c) < radius + r + OBSTACLE_POST_SPACING:
                return False
        for p1, p2 in local_walls:
            if _point_seg_dist(center, p1, p2) < radius + OBSTACLE_POST_SPACING:
                return False

    return True


def _corridor_layout(waypoints, offset, bounce_contact=False):
    n = len(waypoints) - 1
    fwds, perps = [], []
    for i in range(n):
        f, p, _ = _seg_geo(waypoints[i], waypoints[i + 1])
        fwds.append(f)
        perps.append(p)

    start_pts = {1: [], -1: []}
    end_pts = {1: [], -1: []}
    bounce_walls = []

    for i in range(n):
        for side in (1, -1):
            offset_vec = side * offset * perps[i]
            start_pts[side].append(waypoints[i] + offset_vec)
            end_pts[side].append(waypoints[i + 1] + offset_vec)

    for i in range(1, n):
        turn = _cross2(fwds[i - 1], fwds[i])
        outer_side = 1 if turn > 0 else -1
        inner_side = -outer_side
        joint = waypoints[i]

        outer_corner = _line_intersection(
            joint + outer_side * offset * perps[i - 1], fwds[i - 1],
            joint + outer_side * offset * perps[i], fwds[i],
        )
        end_pts[outer_side][i - 1] = outer_corner
        start_pts[outer_side][i] = outer_corner

        bounce_point, bounce_dir = _bounce_line(
            waypoints[i - 1], joint, waypoints[i + 1], contact=bounce_contact
        )
        prev_inner = _line_intersection(
            joint + inner_side * offset * perps[i - 1], fwds[i - 1],
            bounce_point, bounce_dir,
        )
        next_inner = _line_intersection(
            bounce_point, bounce_dir,
            joint + inner_side * offset * perps[i], fwds[i],
        )
        end_pts[inner_side][i - 1] = prev_inner
        start_pts[inner_side][i] = next_inner
        bounce_walls.append((prev_inner, next_inner))

    return start_pts, end_pts, bounce_walls


def _corridor(waypoints):
    n = len(waypoints) - 1
    start_pts, end_pts, bounce_walls = _corridor_layout(waypoints, CORRIDOR_W)

    walls = []
    for i in range(n):
        wall = _wall_from_pts(start_pts[1][i], end_pts[1][i])
        if wall is not None:
            walls.append(wall)
        wall = _wall_from_pts(start_pts[-1][i], end_pts[-1][i])
        if wall is not None:
            walls.append(wall)

    for p1, p2 in bounce_walls:
        wall = _wall_from_pts(p1, p2)
        if wall is not None:
            walls.append(wall)

    wall = _wall_from_pts(start_pts[1][0], start_pts[-1][0])
    if wall is not None:
        walls.append(wall)
    wall = _wall_from_pts(end_pts[1][-1], end_pts[-1][-1])
    if wall is not None:
        walls.append(wall)

    return walls


def _corridor_collision_walls(waypoints):
    offset = CORRIDOR_W - WALL_T - BALL_R
    start_pts, end_pts, bounce_walls = _corridor_layout(
        waypoints, offset, bounce_contact=True
    )
    n = len(waypoints) - 1
    walls = []

    for i in range(n):
        if np.linalg.norm(end_pts[1][i] - start_pts[1][i]) > MIN_WALL_LENGTH:
            walls.append((start_pts[1][i], end_pts[1][i]))
        if np.linalg.norm(end_pts[-1][i] - start_pts[-1][i]) > MIN_WALL_LENGTH:
            walls.append((start_pts[-1][i], end_pts[-1][i]))

    for p1, p2 in bounce_walls:
        if np.linalg.norm(p2 - p1) > MIN_WALL_LENGTH:
            walls.append((p1, p2))

    if np.linalg.norm(start_pts[-1][0] - start_pts[1][0]) > MIN_WALL_LENGTH:
        walls.append((start_pts[1][0], start_pts[-1][0]))
    if np.linalg.norm(end_pts[-1][-1] - end_pts[1][-1]) > MIN_WALL_LENGTH:
        walls.append((end_pts[1][-1], end_pts[-1][-1]))

    return walls


# --- obstacles (must not block the solution path centerline) ---

def _obstacles(rng, waypoints, start, hole, corridor_walls):
    walls = []
    posts = []
    pattern_names = []
    path_nodes = [start] + [np.array(w) for w in waypoints[1:-1]] + [hole]
    path_segments = list(zip(path_nodes[:-1], path_nodes[1:]))
    corridor_segments = [_wall_endpoints(wall) for wall in corridor_walls]
    segment_lengths = [
        np.linalg.norm(np.array(waypoints[i + 1]) - np.array(waypoints[i]))
        for i in range(len(waypoints) - 1)
    ]
    total_path_len = sum(np.linalg.norm(b - a) for a, b in path_segments)
    global_budget = int(np.clip(
        np.round(COURSE_OBSTACLE_BUDGET_BASE + COURSE_OBSTACLE_BUDGET_PER_M * total_path_len),
        2,
        6,
    ))
    target_patterns_per_segment = _distribute_segment_budget(
        rng, segment_lengths, global_budget
    )
    usage_counts = {}
    pattern_builders = [
        _pattern_slalom_posts,
        _pattern_gate_posts,
        _pattern_side_cluster,
        _pattern_staggered_walls,
        _pattern_pinball_pair,
        _pattern_midlane_boxout,
    ]

    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        fwd_hat, perp, seg_len = _seg_geo(a, b)
        target_patterns = target_patterns_per_segment[i]
        if target_patterns == 0:
            continue

        local_walls = []
        local_posts = []
        local_names = []
        placed = 0

        for _ in range(target_patterns * len(pattern_builders)):
            if placed >= target_patterns:
                break
            placed_this_round = False
            for builder in _pattern_priority(rng, pattern_builders, usage_counts, local_names):
                candidate = builder(rng, seg_len)
                if candidate is None:
                    continue
                if not _candidate_valid(candidate, local_walls, local_posts, seg_len,
                                        a, fwd_hat, perp, start, hole, path_segments,
                                        corridor_segments):
                    continue

                local_walls.extend(candidate["walls"])
                local_posts.extend(candidate["posts"])
                local_names.append(candidate["name"])
                pattern_names.append(candidate["name"])
                usage_counts[builder.__name__] = usage_counts.get(builder.__name__, 0) + 1
                placed += 1
                placed_this_round = True
                break
            if not placed_this_round:
                break

        for p1, p2 in local_walls:
            wall = _world_wall(a, fwd_hat, perp, p1, p2)
            if wall is not None:
                walls.append(wall)
        for center, radius in local_posts:
            posts.append(_world_post(a, fwd_hat, perp, center, radius))

    return walls, posts, pattern_names


# --- solution velocity ---

def _compute_solution_velocity(waypoints, start, hole):
    # The ball path is defined exactly by start -> bounce points -> hole.
    path = [start] + [np.array(p) for p in waypoints[1:-1]] + [hole]
    decel = 0.5 * 9.81
    speed_needed = 0.0

    for seg_idx in range(len(path) - 2, -1, -1):
        seg_len = np.linalg.norm(path[seg_idx + 1] - path[seg_idx])
        if seg_idx < len(path) - 2:
            speed_needed /= BOUNCE_RESTITUTION
        speed_needed = np.sqrt(speed_needed ** 2 + 2 * decel * seg_len)

    direction = path[1] - path[0]
    direction /= np.linalg.norm(direction)
    return (direction * speed_needed).tolist()
