// Reinforcement-learning environment: road + traffic + the learning car.
// Exposes reset() / step(action) in the usual Gym style.

const ACTIONS = [
  { label: "↑ gas", forward: true, left: false, right: false, reverse: false },
  {
    label: "↖ gas+left",
    forward: true,
    left: true,
    right: false,
    reverse: false,
  },
  {
    label: "↗ gas+right",
    forward: true,
    left: false,
    right: true,
    reverse: false,
  },
  {
    label: "· coast",
    forward: false,
    left: false,
    right: false,
    reverse: false,
  },
  {
    label: "↓ brake",
    forward: false,
    left: false,
    right: false,
    reverse: true,
  },
];

const STATE_LABELS = [
  "ray 1 (L)",
  "ray 2",
  "ray 3 (mid)",
  "ray 4",
  "ray 5 (R)",
  "speed",
  "angle",
  "lane pos",
];

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const randomBetween = (min, max) => min + Math.random() * (max - min);

class Environment {
  static MAX_STEPS = 3000; // episode time limit
  static STUCK_STEPS = 300; // end episode if no new progress for this long
  static CRASH_PENALTY = -10;

  // traffic settings
  static TRAFFIC_SPRITES = ["car_black", "car_red", "car_yellow"];
  static TRAFFIC_MIN_SPEED = 1.5;
  static TRAFFIC_MAX_SPEED = 2.5;
  static SPAWN_GAP_MIN = 220;
  static SPAWN_GAP_MAX = 420;
  static MIN_LANE_DISTANCE = 150;

  constructor(viewHeight = 800) {
    this.viewHeight = viewHeight;
    this.road = new Road(100, 200);
    this.stateSize = STATE_LABELS.length;
    this.actionCount = ACTIONS.length;
    this.reset();
  }

  reset() {
    const lane = Math.floor(Math.random() * this.road.laneCount);
    this.car = new Car(
      this.road.getLaneCenter(lane),
      100,
      50,
      30,
      "AGENT",
      3,
      "car_blue",
    );
    this.startY = this.car.y;
    this.traffic = [];
    for (let i = 1; i <= 4; i++) this.#spawnTraffic(-150 - i * 200);
    this.nextSpawnY =
      this.car.y -
      randomBetween(Environment.SPAWN_GAP_MIN, Environment.SPAWN_GAP_MAX);

    this.steps = 0;
    this.bestY = this.car.y;
    this.sinceBest = 0;

    // one update so polygons + sensor readings exist before the first state
    this.traffic.forEach((t) => t.update(this.road.borders, []));
    this.car.update(this.road.borders, this.traffic);
    return this.getState();
  }

  getState() {
    const c = this.car;
    const rays = c.sensor.readings.map((r) => (r == null ? 0 : 1 - r.offset));
    return [
      ...rays,
      c.speed / c.maxSpeed,
      clamp(c.angle / (Math.PI / 2), -1, 1),
      clamp((c.x - this.road.x) / (this.road.width / 2), -1, 1),
    ];
  }

  get distance() {
    return Math.max(0, this.startY - this.car.y);
  }

  step(actionIndex) {
    const act = ACTIONS[actionIndex];
    const c = this.car;
    c.controls.forward = act.forward;
    c.controls.left = act.left;
    c.controls.right = act.right;
    c.controls.reverse = act.reverse;

    const prevY = c.y;
    this.#manageTraffic();
    this.traffic.forEach((t) => t.update(this.road.borders, []));
    c.update(this.road.borders, this.traffic);
    this.steps++;

    // ----- reward -----
    let reward = (prevY - c.y) / c.maxSpeed; // forward progress, ~0..1 per step
    if (act.reverse) reward -= 0.05;
    if (act.left || act.right) reward -= 0.01; // discourage wobbling
    const done = c.damaged;
    if (done) reward = Environment.CRASH_PENALTY;

    // ----- time limits (truncation, not a real terminal state) -----
    if (c.y < this.bestY - 1) {
      this.bestY = c.y;
      this.sinceBest = 0;
    } else {
      this.sinceBest++;
    }
    const truncated =
      !done &&
      (this.steps >= Environment.MAX_STEPS ||
        this.sinceBest >= Environment.STUCK_STEPS);

    return { nextState: this.getState(), reward, done, truncated };
  }

  // ---------- traffic ----------
  #carsNear(lane, y) {
    const laneX = this.road.getLaneCenter(lane);
    return this.traffic.filter(
      (t) =>
        Math.abs(t.x - laneX) < 1 &&
        Math.abs(t.y - y) < Environment.MIN_LANE_DISTANCE,
    );
  }

  // spawns one car in a random free lane; always leaves one lane open
  #spawnTraffic(y) {
    const freeLanes = [];
    for (let lane = 0; lane < this.road.laneCount; lane++) {
      if (this.#carsNear(lane, y).length == 0) freeLanes.push(lane);
    }
    if (freeLanes.length <= 1) return;

    const lane = freeLanes[Math.floor(Math.random() * freeLanes.length)];
    const laneX = this.road.getLaneCenter(lane);
    // never spawn slower than cars behind it in the same lane
    const laneSpeeds = this.traffic
      .filter((t) => Math.abs(t.x - laneX) < 1)
      .map((t) => t.maxSpeed);
    const minSpeed = Math.max(Environment.TRAFFIC_MIN_SPEED, ...laneSpeeds);
    const speed = randomBetween(
      minSpeed,
      Math.max(minSpeed, Environment.TRAFFIC_MAX_SPEED),
    );
    const sprites_ = Environment.TRAFFIC_SPRITES;
    const sprite = sprites_[Math.floor(Math.random() * sprites_.length)];
    this.traffic.push(new Car(laneX, y, 50, 30, "DUMMY", speed, sprite));
  }

  #manageTraffic() {
    const car = this.car;
    if (!car.damaged && car.y <= this.nextSpawnY) {
      const spawnY = car.y - this.viewHeight * 0.7 - 100;
      this.#spawnTraffic(spawnY);
      if (Math.random() < 0.25) this.#spawnTraffic(spawnY);
      this.nextSpawnY =
        car.y -
        randomBetween(Environment.SPAWN_GAP_MIN, Environment.SPAWN_GAP_MAX);
    }
    for (let i = this.traffic.length - 1; i >= 0; i--) {
      if (this.traffic[i].y > car.y + this.viewHeight * 0.3 + 200) {
        this.traffic.splice(i, 1);
      }
    }
  }
}
