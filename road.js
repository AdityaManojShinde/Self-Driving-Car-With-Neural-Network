class Road {
  constructor(x, width, laneCount = 3) {
    this.x = x;
    this.width = width;
    this.laneCount = laneCount;

    this.left = x - width / 2;
    this.right = x + width / 2;
    const infinity = 1000000;
    this.top = -infinity;
    this.bottom = infinity;

    const topLeft = { x: this.left, y: this.top };
    const topRight = { x: this.right, y: this.top };
    const bottomLeft = { x: this.left, y: this.bottom };
    const bottomRight = { x: this.right, y: this.bottom };

    this.borders = [
      [topLeft, bottomLeft],
      [topRight, bottomRight],
    ];
  }

  getLaneCenter(laneIndex) {
    const laneWidth = this.width / this.laneCount;
    return (
      this.left +
      laneWidth / 2 +
      Math.min(laneIndex, this.laneCount - 1) * laneWidth
    );
  }

  // viewTop / viewBottom = visible y-range, so we only draw what's on screen
  draw(ctx, viewTop, viewBottom) {
    ctx.lineWidth = 5;
    ctx.strokeStyle = "white";

    // snap to the dash period (40) so the dashes don't shimmer while scrolling
    const start = Math.floor(viewTop / 40) * 40;
    const end = viewBottom;

    for (let i = 1; i <= this.laneCount - 1; i++) {
      const x = lerp(this.left, this.right, i / this.laneCount);
      ctx.setLineDash([20, 20]);
      ctx.beginPath();
      ctx.moveTo(x, start);
      ctx.lineTo(x, end);
      ctx.stroke();
    }

    // solid outer borders
    ctx.setLineDash([]);
    this.borders.forEach((border) => {
      ctx.beginPath();
      ctx.moveTo(border[0].x, start);
      ctx.lineTo(border[1].x, end);
      ctx.stroke();
    });
  }
}
