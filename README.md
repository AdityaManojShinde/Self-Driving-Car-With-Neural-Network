# Self-Driving Car · Deep Q-Network (DQN)

A browser-based self-driving car simulation where an AI agent **learns to drive from scratch using reinforcement learning**. The car starts out knowing nothing, crashes constantly, and gradually learns to stay on the road and dodge traffic. Everything runs locally in your browser with no backend and no build step.

The project started as a neuroevolution demo (random networks, no learning) and was rebuilt around a **Double DQN** agent written with **TensorFlow.js**.

---

## Features

- **Double DQN agent** with replay buffer, target network, and epsilon-greedy exploration
- **Live training dashboard**: stats, network inputs, Q-value per action, and a learning curve
- **Distance sensors**: 10 rays that detect road edges and other cars
- **7 driving actions**: gas, gas+steer, brake, reverse, reverse+steer
- **Endless random traffic** with a guaranteed open lane
- **Play mode**: watch a trained model drive with no time limit, with Pause, Replay and a visible crash
- **Save / Load** in the browser (IndexedDB) and **Download / Load file** for portable models
- **Adjustable simulation speed** (1×–50×) for fast training

---

## Technologies

| Technology                   | Used for                                           |
| ---------------------------- | -------------------------------------------------- |
| **JavaScript (ES2022)**      | All logic: simulation, agent, UI                   |
| **TensorFlow.js 4.22**       | Neural networks, Adam optimizer, training          |
| **HTML5 Canvas**             | Rendering the road, cars, sensors and dashboard    |
| **CSS3**                     | Layout and the control panel                       |
| **IndexedDB / localStorage** | Saving models and training progress in the browser |

No frameworks, bundlers or Node.js are required.

---

## Quick start

### 1. Requirements

- A recent Chrome, Edge or Firefox
- Internet on first load (TensorFlow.js comes from a CDN)
- A car sprite folder named `assets/` next to `index.html`, containing:
  `car_black.png`, `car_blue.png`, `car_red.png`, `car_yellow.png`, `crash.png`

### 2. Run it

Serve the folder with any local web server. Opening `index.html` directly will not work for saving/loading models.

```bash
cd self-driving-dqn
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

(VS Code's _Live Server_ extension also works.)

### 3. Offline use (optional)

```bash
npm i @tensorflow/tfjs@4.22.0
```

Copy `node_modules/@tensorflow/tfjs/dist/tf.min.js` into the project and change the `<script>` tag in `index.html` to point to it.

---

## How to use

The window is split in two: the **simulation** on the left and the **dashboard + controls** on the right.

| Control                  | What it does                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------- |
| **▶ Play / ■ Exit play** | Showcase run: greedy policy, no learning, no time limit, stops on crash            |
| **Pause / Resume**       | Freeze or continue the simulation                                                  |
| **Replay**               | (Play mode) Start a new run                                                        |
| **Mode: TRAIN / WATCH**  | TRAIN explores and learns. WATCH uses only what was learned                        |
| **Sim speed slider**     | How many simulation steps run per frame (1×–50×)                                   |
| **Save / Load**          | Store or restore the model in the browser (IndexedDB). Autosaves every 25 episodes |
| **Download**             | Export `car-dqn.json` + `car-dqn.weights.bin`                                      |
| **Load file**            | Load a downloaded model (select both files together)                               |
| **Reset agent**          | Throw away the model and start training from scratch                               |

### Typical workflow

1. Leave the mode on **TRAIN** and set the speed to about 10–30×.
2. Watch the orange **avg20** line on the chart. When it climbs, the agent is learning.
3. **Download** the model when it performs well.
4. Later: **Load file**, then **▶ Play** to watch it drive.

### Rough training timeline

These are estimates, not measurements.

| Episodes | Behavior                            |
| -------- | ----------------------------------- |
| 0–30     | Random flailing, crashes in seconds |
| 30–150   | Stays on the road                   |
| 150–500  | Starts swerving around traffic      |
| 500–2000 | Reliably dodges most cars           |

---

## How it works

### The environment

A Gym-style wrapper (`env.js`) with `reset()` and `step(action)`.

**State (13 inputs)**

- 10 sensor rays (`1 – distance`, so closer obstacles give larger values)
- Speed (normalised)
- Steering angle (normalised)
- Position across the road (normalised)

**Actions (7)**

| #   | Action                              |
| --- | ----------------------------------- |
| 0   | ↑ gas                               |
| 1   | ↖ gas + left                        |
| 2   | ↗ gas + right                       |
| 3   | ■ brake (slow to 0, never reverses) |
| 4   | ↓ reverse                           |
| 5   | ↙ reverse + left                    |
| 6   | ↘ reverse + right                   |

**Reward**

- Forward progress each step (about 0 to 1)
- -0.01 for steering, to discourage wobbling
- Backward movement gives negative reward through negative progress
- **-10** on a crash

**Episode ends when**

- the car crashes (terminal), or
- 3000 steps pass, or the car makes no new progress for 300 steps (time-limit endings, not treated as failures)

In Play mode the time limits are disabled.

### The agent

- **Network:** 13 → 64 → 64 → 7 (ReLU hidden layers, linear output of Q-values)
- **Algorithm:** Double DQN, where the online network picks the next action and the target network scores it
- **Loss / optimizer:** Huber loss, Adam
- **Replay buffer:** 50,000 transitions, minibatch of 64
- **Exploration:** epsilon decays from 1.0 to 0.05 over 40,000 steps

| Hyperparameter  | Default     |
| --------------- | ----------- |
| `gamma`         | 0.98        |
| `learningRate`  | 0.001       |
| `batchSize`     | 64          |
| `bufferSize`    | 50,000      |
| `warmup`        | 1,000 steps |
| `trainEvery`    | 4 steps     |
| `targetUpdate`  | 1,000 steps |
| `epsDecaySteps` | 40,000      |

Change them in the `DQNAgent` constructor (`dqn.js`).

### Collisions

Cars collide using a hitbox that is **80% of their drawn size** (`Car.HITBOX_SCALE`), giving a bit of forgiveness. The sensor rays still hit the full-size shape.

---

## Project structure

```
self-driving-dqn/
├── index.html        Page layout, control panel, script loading
├── style.css         Styles (grass background, panel)
├── main.js           App loop, UI wiring, train / watch / play modes
├── env.js            Environment: reset(), step(), reward, traffic, actions
├── dqn.js            Replay buffer + Double DQN agent (TensorFlow.js)
├── car.js            Car physics, hitbox, drawing, crash sprite
├── sensor.js         Ray casting and distance readings (RAY_COUNT)
├── road.js           Road geometry and drawing
├── controls.js       Control flags (forward, left, right, reverse, brake)
├── visualizer.js     Dashboard: stats, state bars, Q-values, chart
├── sprites.js        Image loader for the car and crash sprites
├── utils.js          lerp, line intersection, polygon intersection
└── assets/           car_*.png and crash.png (you provide these)
```

### Script load order

`sprites.js` → `utils.js` → `visualizer.js` → `dqn.js` → `sensor.js` → `road.js` → `controls.js` → `car.js` → `env.js` → `main.js`

(after TensorFlow.js from the CDN).

---

## Customising

| Want to change            | Edit                                                |
| ------------------------- | --------------------------------------------------- |
| Number of sensor rays     | `RAY_COUNT` in `sensor.js`                          |
| Road width                | `Environment.ROAD_WIDTH` in `env.js` (default 300)  |
| Collision forgiveness     | `Car.HITBOX_SCALE` in `car.js` (default 0.8)        |
| Episode time limits       | `MAX_STEPS`, `STUCK_STEPS` in `env.js`              |
| Crash penalty             | `CRASH_PENALTY` in `env.js`                         |
| Traffic speed and density | the `TRAFFIC_*` and `SPAWN_*` constants in `env.js` |
| Network size              | `hidden` in the `DQNAgent` config (`dqn.js`)        |
| TensorFlow backend        | `BACKEND` in `main.js` (`"cpu"` or `"webgl"`)       |

> **Note:** Changing the number of rays, the actions, or the state inputs changes the network's shape. Saved models will no longer load, and you need to **Reset agent** and retrain.

---

## Tips and limitations

- Learning is driven by total **steps**, not episodes. Early episodes are very short.
- The sensors only look forward, so the car is blind when reversing.
- The agent rarely brakes because steering around cars is cheaper than slowing down under the current reward.
- Training can drift. Download good checkpoints at several points.
- The replay buffer is not saved. After loading a model, the first 1,000 steps only collect experience before learning resumes.
- Models saved to the browser belong to that browser and address (for example `localhost:8000`).
