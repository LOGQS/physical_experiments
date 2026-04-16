# Golf Sim-to-Sim System Identification

## Origin

Inspired by [Stuff Made Here's self-aiming golf club](https://youtu.be/2OfjZ3ORJfc). He tried tuning physics parameters by hand after his stochastic solver failed (malicious compliance — minimized error by stopping the ball entirely). Couldn't find his code, so this is an imitation based on the video.

## Problem

Given a "reality" simulator with hidden parameters, infer them from observed trajectories with known initial conditions and actions.

- **Reality:** MuJoCo golf env with hidden ground-truth parameters + noise
- **Model:** Your system's belief about reality
- **Goal:** Minimize trajectory prediction error on held-out rollouts

Note: multiple parameter sets may produce similar trajectories (non-identifiability). Success means finding parameters that generalize, not necessarily recovering ground truth exactly.

## Constraints

Sim-to-sim because no hardware access. Both sides run MuJoCo — the model class is assumed correct. Real sim-to-real additionally involves unmodeled phenomena outside your model class.