const carCanvas = document.getElementById("car_canvas");
const networkCanvas = document.getElementById("neural_network");
const carCtx = carCanvas.getContext("2d");
const networkCtx = networkCanvas.getContext("2d");

// "cpu" is usually faster than "webgl" for tiny networks (no GPU read-back stalls).
// Change to "webgl" if you scale the network up a lot.
const BACKEND = "cpu";

let env, agent, state;
let trainMode = true; // true: explore + learn | false: greedy, no learning
let playMode = false; // showcase run: greedy, no time limits, stops on crash
let paused = false;
let crashed = false; // only used in play mode
let stepsPerFrame = 5;
let savedSpeed = 5; // slider value to restore when leaving play mode
let pauseFrames = 0; // short pause after a crash when running at 1x (training)
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

// start a fresh run (new car + traffic)
function restartRun() {
  state = env.reset();
  epReward = 0;
  crashed = false;
}

function setSpeed(v) {
  stepsPerFrame = v;
  $("speed").value = v;
  $("speedVal").textContent = v + "×";
}

function enterPlay() {
  playMode = true;
  savedSpeed = stepsPerFrame;
  setSpeed(1);
  env.limitsEnabled = false; // no 3000-step / stuck limits
  paused = false;
  $("pauseBtn").textContent = "Pause";
  $("playBtn").textContent = "■ Exit play";
  $("replayBtn").disabled = false;
  $("modeBtn").disabled = true;
  restartRun();
  setStatus("Play mode: greedy policy, no time limit. Runs until it crashes.");
}

function exitPlay() {
  playMode = false;
  env.limitsEnabled = true;
  setSpeed(savedSpeed);
  paused = false;
  $("pauseBtn").textContent = "Pause";
  $("playBtn").textContent = "▶ Play";
  $("replayBtn").disabled = true;
  $("modeBtn").disabled = false;
  restartRun();
  setStatus("Back to " + (trainMode ? "training." : "watch mode."));
}

$("playBtn").addEventListener("click", () =>
  playMode ? exitPlay() : enterPlay(),
);
$("pauseBtn").addEventListener("click", () => {
  paused = !paused;
  $("pauseBtn").textContent = paused ? "Resume" : "Pause";
});
$("replayBtn").addEventListener("click", () => {
  paused = false;
  $("pauseBtn").textContent = "Pause";
  restartRun();
});

$("speed").addEventListener("input", (e) => {
  setSpeed(Number(e.target.value));
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
    restartRun();
    setStatus("Model loaded (" + agent.totalSteps + " steps of experience).");
  } catch (e) {
    setStatus("Load failed: " + (e.message || "no saved model found."));
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
    restartRun();
    setStatus("Model loaded from files. Press Play to watch it drive.");
  } catch (err) {
    setStatus("Load failed: " + err.message);
  }
});
$("resetBtn").addEventListener("click", () => {
  agent.dispose();
  agent = new DQNAgent(env.stateSize, env.actionCount);
  history.length = 0;
  restartRun();
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
  restartRun();
}

function stepOnce() {
  if (playMode) {
    // showcase: greedy action, never learns, stops (frozen) when it crashes
    const action = agent.act(state, false);
    const { nextState, reward, done } = env.step(action);
    state = nextState;
    epReward += reward;
    if (done) crashed = true;
    return;
  }

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

function drawOverlay(title, subtitle) {
  const cx = carCanvas.width / 2;
  const cy = carCanvas.height * 0.3;
  carCtx.save();
  carCtx.fillStyle = "rgba(0,0,0,0.6)";
  carCtx.fillRect(cx - 170, cy - 45, 340, 90);
  carCtx.textAlign = "center";
  carCtx.textBaseline = "middle";
  carCtx.fillStyle = "white";
  carCtx.font = "bold 28px Arial";
  carCtx.fillText(title, cx, cy - 12);
  carCtx.font = "15px Arial";
  carCtx.fillStyle = "#ddd";
  carCtx.fillText(subtitle, cx, cy + 20);
  carCtx.restore();
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
  car.draw(carCtx, "blue"); // draws the crash sprite once damaged
  carCtx.restore();

  if (crashed) {
    drawOverlay(
      "CRASHED",
      Math.round(env.distance) + " px driven · press Replay",
    );
  } else if (paused) {
    drawOverlay("PAUSED", "press Resume to continue");
  }

  const q = agent.qValues(state);
  const distances = history.map((h) => h.distance);
  const recent = distances.slice(-20);
  const avg = recent.length
    ? recent.reduce((a, b) => a + b, 0) / recent.length
    : 0;
  const best = distances.length ? Math.max(...distances) : 0;

  const stats = playMode
    ? [
        "mode         PLAY (greedy, no time limit)",
        "status       " + (crashed ? "crashed" : paused ? "paused" : "driving"),
        "steps        " + env.steps,
        "distance     " + Math.round(env.distance) + " px",
        "speed        " + (car.speed * 60).toFixed(0) + " px/s",
        "ep. reward   " + epReward.toFixed(1),
      ]
    : [
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
      ];

  Visualizer.draw(networkCtx, {
    stats,
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
  } else if (!paused && !crashed) {
    for (let i = 0; i < stepsPerFrame && !crashed; i++) stepOnce();
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
