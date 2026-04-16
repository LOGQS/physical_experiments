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
MAX_BOUNCES = 8
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


def generate_course(seed=None, n_bounces=None):
    rng = np.random.default_rng(seed)
    if n_bounces is None:
        target_bounces = int(rng.integers(0, MAX_BOUNCES + 1))
    else:
        target_bounces = int(n_bounces)
        if not 0 <= target_bounces <= MAX_BOUNCES:
            raise ValueError(f"n_bounces must be in [0, {MAX_BOUNCES}]")

    for _ in range(200):
        waypoints = _make_bounce_path(rng, target_bounces)
        if waypoints is None or not _path_valid(waypoints):
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
        "solution_velocity": solution_vel,
    }


# --- path ---

def _make_bounce_path(rng, n_bounces):
    bound = ARENA - CORRIDOR_W - 0.15
    n_segments = n_bounces + 1

    # Choose a total forward span that stays feasible inside the arena while
    # still keeping low-bounce layouts long enough to read like real holes.
    min_total = 2.0 + 0.15 * max(0, n_bounces - 2)
    max_total = min(4.2, 2.8 + 0.28 * n_bounces)
    min_dx = max(0.24, min_total / n_segments)
    max_dx = min(2.2, max_total / n_segments)
    if min_dx > max_dx:
        return None

    for _ in range(200):
        theta = rng.uniform(0, 2 * np.pi)
        fwd = np.array([np.cos(theta), np.sin(theta)])
        perp = np.array([-fwd[1], fwd[0]])

        dx_base = rng.uniform(min_dx, max_dx)
        dx_steps = rng.uniform(0.85, 1.15, size=n_segments)
        dx_steps *= (dx_base * n_segments) / dx_steps.sum()

        if n_bounces == 0:
            y_vals = np.zeros(2)
        else:
            amp_base = rng.uniform(0.32 * dx_base, 0.65 * dx_base)
            first_side = rng.choice([-1, 1])
            y_vals = [0.0]
            for i in range(n_bounces):
                side = first_side * ((-1) ** i)
                amp = amp_base * rng.uniform(0.8, 1.2)
                y_vals.append(side * amp)
            y_vals.append(0.0)
            y_vals = np.array(y_vals)

        x_vals = np.concatenate(([0.0], np.cumsum(dx_steps)))
        x_vals -= x_vals[-1] / 2

        local_pts = [x * fwd + y * perp for x, y in zip(x_vals, y_vals)]
        pts = np.array(local_pts)

        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        if np.any(maxs - mins > 2 * bound):
            continue

        shift_lo = -bound - mins
        shift_hi = bound - maxs
        shift = rng.uniform(shift_lo, shift_hi)
        pts = pts + shift

        pts = [np.array(p) for p in pts]
        if not _segments_ok(pts):
            continue
        if not _angles_ok(pts):
            continue
        return pts

    return None


def _segments_ok(pts):
    for i in range(len(pts) - 1):
        d = np.linalg.norm(pts[i + 1] - pts[i])
        if d < 0.3 or d > 2.5:
            return False
    return True


def _angles_ok(pts):
    for i in range(1, len(pts) - 1):
        d_in = pts[i] - pts[i - 1]
        d_out = pts[i + 1] - pts[i]
        cos_a = np.dot(d_in, d_out) / (np.linalg.norm(d_in) * np.linalg.norm(d_out))
        if cos_a > 0.5 or cos_a < -0.5:
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
