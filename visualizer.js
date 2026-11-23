// Training dashboard drawn on the right-hand canvas:
// stats, network input (state), Q-values per action, and a learning curve.
class Visualizer {
  static draw(ctx, info) {
    const W = ctx.canvas.width;
    const H = ctx.canvas.height;
    const m = 30;
    const w = W - m * 2;
    let y = m;

    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillStyle = "white";
    ctx.font = "bold 20px Arial";
    ctx.fillText("Deep Q-Network · TensorFlow.js", m, y);
    y += 36;

    ctx.font = "14px monospace";
    ctx.fillStyle = "#cfd8ff";
    for (const line of info.stats) {
      ctx.fillText(line, m, y);
      y += 20;
    }
    y += 12;

    y = Visualizer.#drawBars(
      ctx,
      "STATE (network input)",
      info.stateLabels,
      info.state,
      m,
      y,
      w,
      -1,
      1,
      -1,
    );
    y += 14;

    const qmax = Math.max(1, ...info.q.map((v) => Math.abs(v)));
    y = Visualizer.#drawBars(
      ctx,
      "Q-VALUES (one per action, highlighted = greedy choice)",
      info.actionLabels,
      info.q,
      m,
      y,
      w,
      -qmax,
      qmax,
      info.greedyAction,
    );
    y += 14;

    const chartH = Math.max(60, H - y - 150); // bottom space is for the HTML panel
    Visualizer.#drawChart(
      ctx,
      "DISTANCE PER EPISODE",
      info.series,
      m,
      y,
      w,
      chartH,
    );
  }

  static #drawBars(ctx, title, labels, values, x, y, w, min, max, highlight) {
    ctx.font = "bold 13px Arial";
    ctx.fillStyle = "#8fa0ff";
    ctx.fillText(title, x, y);
    y += 22;

    const labelW = 110;
    const valW = 64;
    const barX = x + labelW;
    const barW = w - labelW - valW;
    const rowH = 20;
    const zeroX = barX + (barW * (0 - min)) / (max - min);

    for (let i = 0; i < values.length; i++) {
      const hl = i == highlight;
      ctx.font = "13px monospace";
      ctx.fillStyle = hl ? "#ffd54a" : "#cfd8ff";
      ctx.fillText(labels[i], x, y + 2);

      ctx.fillStyle = "rgba(255,255,255,0.08)";
      ctx.fillRect(barX, y, barW, rowH - 4);

      const v = Math.max(min, Math.min(max, values[i]));
      const vx = barX + (barW * (v - min)) / (max - min);
      ctx.fillStyle = hl ? "#ffd54a" : v >= 0 ? "#00ff88" : "#4d9dff";
      ctx.fillRect(Math.min(zeroX, vx), y, Math.abs(vx - zeroX), rowH - 4);

      ctx.fillStyle = hl ? "#ffd54a" : "#cfd8ff";
      ctx.fillText(values[i].toFixed(2), barX + barW + 8, y + 2);
      y += rowH;
    }
    return y;
  }

  static #drawChart(ctx, title, series, x, y, w, h) {
    ctx.font = "bold 13px Arial";
    ctx.fillStyle = "#8fa0ff";
    ctx.fillText(title, x, y);
    y += 22;
    h -= 22;

    ctx.fillStyle = "rgba(255,255,255,0.06)";
    ctx.fillRect(x, y, w, h);

    let maxVal = 1;
    let maxLen = 2;
    for (const s of series) {
      for (const v of s.data) maxVal = Math.max(maxVal, v);
      maxLen = Math.max(maxLen, s.data.length);
    }

    for (const s of series) {
      if (s.data.length < 2) continue;
      ctx.beginPath();
      ctx.lineWidth = 2;
      ctx.strokeStyle = s.color;
      s.data.forEach((v, i) => {
        const px = x + (i / (maxLen - 1)) * w;
        const py = y + h - (v / maxVal) * (h - 6) - 3;
        i == 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    ctx.font = "12px monospace";
    ctx.fillStyle = "#cfd8ff";
    ctx.fillText(Math.round(maxVal) + " px", x + 6, y + 4);
    let lx = x + w - 6;
    ctx.textAlign = "right";
    for (const s of series) {
      ctx.fillStyle = s.color;
      ctx.fillText("■ " + s.name, lx, y + 4);
      lx -= 130;
    }
    ctx.textAlign = "left";
  }
}
