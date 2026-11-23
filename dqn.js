// Deep Q-Network (Double DQN) built on TensorFlow.js
// Replaces the old random/neuroevolution network (network.js).

const MODEL_URL = "indexeddb://car-dqn";
const META_KEY = "car-dqn-meta";

class ReplayBuffer {
  constructor(capacity, stateSize) {
    this.capacity = capacity;
    this.stateSize = stateSize;
    this.states = new Float32Array(capacity * stateSize);
    this.nextStates = new Float32Array(capacity * stateSize);
    this.actions = new Int32Array(capacity);
    this.rewards = new Float32Array(capacity);
    this.dones = new Float32Array(capacity);
    this.size = 0;
    this.pos = 0;
  }

  add(s, a, r, ns, done) {
    this.states.set(s, this.pos * this.stateSize);
    this.nextStates.set(ns, this.pos * this.stateSize);
    this.actions[this.pos] = a;
    this.rewards[this.pos] = r;
    this.dones[this.pos] = done ? 1 : 0;
    this.pos = (this.pos + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
  }

  sample(n) {
    const S = this.stateSize;
    const states = new Float32Array(n * S);
    const nextStates = new Float32Array(n * S);
    const actions = new Int32Array(n);
    const rewards = new Float32Array(n);
    const dones = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const k = Math.floor(Math.random() * this.size);
      states.set(this.states.subarray(k * S, (k + 1) * S), i * S);
      nextStates.set(this.nextStates.subarray(k * S, (k + 1) * S), i * S);
      actions[i] = this.actions[k];
      rewards[i] = this.rewards[k];
      dones[i] = this.dones[k];
    }
    return { states, nextStates, actions, rewards, dones };
  }
}

class DQNAgent {
  constructor(stateSize, actionCount, config = {}) {
    this.stateSize = stateSize;
    this.actionCount = actionCount;
    this.cfg = {
      hidden: [64, 64],
      gamma: 0.98,
      learningRate: 0.001,
      batchSize: 64,
      bufferSize: 50000,
      warmup: 1000, // transitions to collect before learning starts
      trainEvery: 4, // env steps between gradient updates
      targetUpdate: 1000, // env steps between target-network syncs
      epsStart: 1.0,
      epsEnd: 0.05,
      epsDecaySteps: 40000,
      ...config,
    };

    this.online = this.#buildModel();
    this.target = this.#buildModel();
    this.syncTarget();
    this.optimizer = tf.train.adam(this.cfg.learningRate);
    this.buffer = new ReplayBuffer(this.cfg.bufferSize, stateSize);

    this.totalSteps = 0;
    this.episodes = 0;
    this.lastLoss = null;
  }

  #buildModel() {
    const model = tf.sequential();
    this.cfg.hidden.forEach((units, i) => {
      model.add(
        tf.layers.dense({
          units,
          activation: "relu",
          ...(i == 0 ? { inputShape: [this.stateSize] } : {}),
        }),
      );
    });
    model.add(tf.layers.dense({ units: this.actionCount })); // linear Q-values
    return model;
  }

  get epsilon() {
    const t = Math.min(1, this.totalSteps / this.cfg.epsDecaySteps);
    return lerp(this.cfg.epsStart, this.cfg.epsEnd, t);
  }

  syncTarget() {
    this.target.setWeights(this.online.getWeights());
  }

  qValues(state) {
    return tf.tidy(() =>
      Array.from(this.online.predict(tf.tensor2d([state])).dataSync()),
    );
  }

  // epsilon-greedy when explore=true, purely greedy otherwise
  act(state, explore = true) {
    if (explore && Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actionCount);
    }
    const q = this.qValues(state);
    return q.indexOf(Math.max(...q));
  }

  // store a transition and (when it's time) learn from a random minibatch
  step(state, action, reward, nextState, done) {
    this.buffer.add(state, action, reward, nextState, done);
    this.totalSteps++;

    if (
      this.buffer.size >= this.cfg.warmup &&
      this.totalSteps % this.cfg.trainEvery == 0
    ) {
      this.lastLoss = this.#trainBatch();
    }
    if (this.totalSteps % this.cfg.targetUpdate == 0) {
      this.syncTarget();
    }
  }

  #trainBatch() {
    const { batchSize: B, gamma } = this.cfg;
    const A = this.actionCount;
    const b = this.buffer.sample(B);
    const varList = this.online.trainableWeights.map((w) => w.val);

    const lossTensor = tf.tidy(() => {
      const s = tf.tensor2d(b.states, [B, this.stateSize]);
      const ns = tf.tensor2d(b.nextStates, [B, this.stateSize]);
      const a = tf.tensor1d(b.actions, "int32");
      const r = tf.tensor1d(b.rewards);
      const d = tf.tensor1d(b.dones);

      // Double DQN: online net picks the best next action, target net scores it
      const bestNext = this.online.predict(ns).argMax(1);
      const nextQ = this.target
        .predict(ns)
        .mul(tf.oneHot(bestNext, A).cast("float32"))
        .sum(1);
      const targetQ = r.add(nextQ.mul(gamma).mul(tf.scalar(1).sub(d)));

      const lossFn = () => {
        const q = this.online
          .predict(s)
          .mul(tf.oneHot(a, A).cast("float32"))
          .sum(1);
        return tf.losses.huberLoss(targetQ, q);
      };
      return this.optimizer.minimize(lossFn, true, varList);
    });

    const loss = lossTensor.dataSync()[0];
    lossTensor.dispose();
    return loss;
  }

  async save() {
    await this.online.save(MODEL_URL);
    localStorage.setItem(
      META_KEY,
      JSON.stringify({ totalSteps: this.totalSteps, episodes: this.episodes }),
    );
  }

  async download() {
    await this.online.save("downloads://car-dqn");
  }

  async load() {
    const model = await tf.loadLayersModel(MODEL_URL);
    this.online.dispose();
    this.online = model;
    this.syncTarget();
    const meta = JSON.parse(localStorage.getItem(META_KEY) || "{}");
    this.totalSteps = meta.totalSteps || 0;
    this.episodes = meta.episodes || 0;
    this.buffer = new ReplayBuffer(this.cfg.bufferSize, this.stateSize);
  }

  // files: the car-dqn.json + car-dqn.weights.bin picked by the user
  async loadFromFiles(files) {
    const json = files.find((f) => f.name.endsWith(".json"));
    const bin = files.find((f) => f.name.endsWith(".bin"));
    if (!json || !bin) throw new Error("Select both the .json and .bin files");
    const model = await tf.loadLayersModel(tf.io.browserFiles([json, bin]));
    if (model.inputs[0].shape[1] != this.stateSize) {
      model.dispose();
      throw new Error("Model input size doesn't match this project");
    }
    this.online.dispose();
    this.online = model;
    this.syncTarget();
    // step count isn't stored in the files: start with low exploration
    this.totalSteps = this.cfg.epsDecaySteps;
    this.buffer = new ReplayBuffer(this.cfg.bufferSize, this.stateSize);
  }

  dispose() {
    this.online.dispose();
    this.target.dispose();
    this.optimizer.dispose();
  }
}
