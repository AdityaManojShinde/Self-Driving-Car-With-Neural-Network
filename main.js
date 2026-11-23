const carCanvas = document.getElementById("car_canvas");
const networkCanvas = document.getElementById("neural_network");
const carCtx = carCanvas.getContext("2d");
const networkCtx = networkCanvas.getContext("2d");

// "cpu" is usually faster than "webgl" for tiny networks (no GPU read-back stalls).
// Change to "webgl" if you scale the network up a lot.
const BACKEND = "cpu";

let env, agent, state;
let trainMode = true; // true: explore + learn | false: greedy, no learning
let stepsPerFrame = 5;
let pauseFrames = 0; // short pause after a crash when running at 1x
let epReward = 0;
const history = []; // { distance, reward } per finished episode
const AUTOSAVE_EVERY = 25; // episodes

function resizeCanvases() {
  carCanvas.width = window.innerWidth / 2;
  networkCanvas.width = window.innerWidth / 2;
  carCanvas.height = window.innerHeight;
  networkCanvas.height = window.innerHeight;
  if (env) env.viewHeight = carCanvas.height;
}
window.addEventListener("resize", resizeCanvases);

// ---------- UI ----------
const $ = (id) => document.getElementById(id);
function setStatus(msg) {
  $("status").textContent = msg;
}

$("speed").addEventListener("input", (e) => {
  stepsPerFrame = Number(e.target.value);
  $("speedVal").textContent = stepsPerFrame + "×";
});
$("modeBtn").addEventListener("click", () => {
  trainMode = !trainMode;
  $("modeBtn").textContent = trainMode ? "Mode: TRAIN" : "Mode: WATCH (greedy)";
});
$("saveBtn").addEventListener("click", async () => {
  await agent.save();
  setStatus("Saved to browser storage (IndexedDB).");
});
$("loadBtn").addEventListener("click", async () => {
  try {
    await agent.load();
    state = env.reset();
    epReward = 0;
    setStatus("Model loaded (" + agent.totalSteps + " steps of experience).");
  } catch (e) {
    setStatus("No saved model found.");
  }
});
$("downloadBtn").addEventListener("click", async () => {
  await agent.download();
  setStatus("Downloaded car-dqn.json + car-dqn.weights.bin");
});
$("loadFileBtn").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", async (e) => {
  const files = Array.from(e.target.files);
  e.target.value = ""; // allow picking the same files again
  if (!files.length) return;
  try {
    await agent.loadFromFiles(files);
    state = env.reset();
    epReward = 0;
    setStatus("Model loaded from files. Switch to WATCH to see it drive.");
  } catch (err) {
    setStatus("Load failed: " + err.message);
  }
});
$("resetBtn").addEventListener("click", () => {
  agent.dispose();
  agent = new DQNAgent(env.stateSize, env.actionCount);
  history.length = 0;
  state = env.reset();
  epReward = 0;
  setStatus("Agent reset.");
});

// ---------- training loop ----------
function endEpisode() {
  history.push({ distance: env.distance, reward: epReward });
  if (history.length > 200) history.shift();
  agent.episodes++;
  if (trainMode && agent.episodes % AUTOSAVE_EVERY == 0) {
    agent
      .save()
      .then(() => setStatus("Autosaved at episode " + agent.episodes));
  }
  if (stepsPerFrame == 1) pauseFrames = 30;
  state = env.reset();
  epReward = 0;
}

function stepOnce() {
  const action = agent.act(state, trainMode);
  const { nextState, reward, done, truncated } = env.step(action);
  if (trainMode) {
    // only a real crash is terminal; time-limit endings still bootstrap
    agent.step(state, action, reward, nextState, done);
  }
  state = nextState;
  epReward += reward;
  if (done || truncated) endEpisode();
}

function movingAverage(values, window = 20) {
  return values.map((_, i) => {
    const slice = values.slice(Math.max(0, i - window + 1), i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

function draw() {
  carCtx.clearRect(0, 0, carCanvas.width, carCanvas.height);
  networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);

  const { car, traffic, road } = env;
  const offsetY = -car.y + carCanvas.height * 0.7;
  carCtx.save();
  carCtx.translate(carCanvas.width / 2 - road.x, offsetY);
  road.draw(carCtx, -offsetY - 50, -offsetY + carCanvas.height + 50);
  for (let i = 0; i < traffic.length; i++) traffic[i].draw(carCtx, "red");
  car.draw(carCtx, "blue");
  carCtx.restore();

  const q = agent.qValues(state);
  const distances = history.map((h) => h.distance);
  const recent = distances.slice(-20);
  const avg = recent.length
    ? recent.reduce((a, b) => a + b, 0) / recent.length
    : 0;
  const best = distances.length ? Math.max(...distances) : 0;

  Visualizer.draw(networkCtx, {
    stats: [
      "episode      " + agent.episodes + (trainMode ? "" : "  (watching)"),
      "total steps  " + agent.totalSteps,
      "epsilon      " + (trainMode ? agent.epsilon.toFixed(3) : "0 (greedy)"),
      "replay       " + agent.buffer.size + " / " + agent.cfg.bufferSize,
      "loss         " +
        (agent.lastLoss == null ? "-" : agent.lastLoss.toFixed(4)),
      "distance     " +
        Math.round(env.distance) +
        " px   best " +
        Math.round(best) +
        "   avg20 " +
        Math.round(avg),
      "ep. reward   " + epReward.toFixed(1),
    ],
    stateLabels: STATE_LABELS,
    state,
    actionLabels: ACTIONS.map((a) => a.label),
    q,
    greedyAction: q.indexOf(Math.max(...q)),
    series: [
      { name: "episode", color: "#4d9dff", data: distances },
      { name: "avg20", color: "#ffb74d", data: movingAverage(distances) },
    ],
  });
}

function animate() {
  if (pauseFrames > 0) {
    pauseFrames--;
  } else {
    for (let i = 0; i < stepsPerFrame; i++) stepOnce();
  }
  draw();
  requestAnimationFrame(animate);
}

// ---------- start ----------
(async function init() {
  await tf.ready();
  await tf.setBackend(BACKEND);
  resizeCanvases();
  env = new Environment(carCanvas.height);
  agent = new DQNAgent(env.stateSize, env.actionCount);
  state = env.reset();
  setStatus("TensorFlow.js backend: " + tf.getBackend());
  requestAnimationFrame(animate);
})();
