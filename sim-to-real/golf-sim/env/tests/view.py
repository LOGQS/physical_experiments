import sys
import time
import ctypes
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "env" / "src"))

from env.src.golf import GolfEnv, BALL_RADIUS

if sys.platform == "win32":
    _user32 = ctypes.windll.user32
    _SW_MAXIMIZE = 3

    def _maximize_foreground_window(retries=20, delay=0.05):
        for _ in range(retries):
            hwnd = _user32.GetForegroundWindow()
            if hwnd:
                _user32.ShowWindow(hwnd, _SW_MAXIMIZE)
                return
            time.sleep(delay)

seed = int(sys.argv[1]) if len(sys.argv) > 1 else None

print("Controls:")
print("  SPACE       — hit (random direction/speed)")
print("  S           — compute & play solution (single shot)")
print("  P           — pause / unpause")
print("  R           — new random course (reopens window)")
print("  Right-drag  — pan camera")
print("  Left-drag   — rotate camera")
print("  Scroll      — zoom")

running = True
while running:
    env = GolfEnv(seed=seed)
    cached_solution = None

    paused = True
    hit_pending = False
    solve_pending = False
    new_course = False

    def on_key(keycode):
        global hit_pending, paused, new_course, solve_pending
        if keycode == 32:    # SPACE
            hit_pending = True
        elif keycode == 83:  # S
            solve_pending = True
        elif keycode == 80:  # P
            paused = not paused
        elif keycode == 82:  # R
            new_course = True

    def place_ball(pos):
        env.reset_ball(pos)

    def apply_hit(vx, vy):
        env.set_velocity(vx, vy)

    place_ball(env.ball_start)

    print(f"\nCourse seed: {seed}")
    print(f"Ball: ({env.ball_start[0]:.2f}, {env.ball_start[1]:.2f})  "
          f"Hole: ({env.hole_pos[0]:.2f}, {env.hole_pos[1]:.2f})  "
          f"Bounces: {env.n_bounces}  Segments: {len(env.waypoints)-1}")

    with mujoco.viewer.launch_passive(env.model, env.data,
                                       key_callback=on_key) as v:
        _maximize_foreground_window()
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.lookat[:] = [0, 0, 0]
        v.cam.distance = 6.0
        v.cam.elevation = -55
        v.cam.azimuth = 135

        wall_start = time.time()
        sim_start = env.data.time

        while v.is_running():
            if new_course:
                seed = None
                break

            if solve_pending:
                solve_pending = False
                if cached_solution is None:
                    cached_solution = env.solve()
                    print("  Solution ready")
                place_ball(env.ball_start)
                vx, vy = cached_solution
                apply_hit(vx, vy)
                paused = False
                wall_start = time.time()
                sim_start = env.data.time

            if hit_pending:
                hit_pending = False
                place_ball(env.ball_start)
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(1.5, 4.0)
                apply_hit(speed * np.cos(angle), speed * np.sin(angle))
                paused = False
                wall_start = time.time()
                sim_start = env.data.time

            if paused:
                v.sync()
                time.sleep(0.01)
                continue

            env.step()
            v.sync()

            ball_xy = env.data.qpos[env._qpos_adr:env._qpos_adr + 2]
            spd = np.linalg.norm(env.data.qvel[env._qvel_adr:env._qvel_adr + 2])
            if env.ball_in_hole(ball_xy) and spd < 0.5:
                print("  Ball in hole!")

            sim_elapsed = env.data.time - sim_start
            wall_elapsed = time.time() - wall_start
            dt = sim_elapsed - wall_elapsed
            if dt > 0:
                time.sleep(dt)

        if not new_course:
            running = False
