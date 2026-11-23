class Car {
  // controlType: "KEYS" (human), "DUMMY" (traffic), "AGENT" (driven by the DQN via env.step)
  constructor(
    x,
    y,
    height,
    width,
    controlType,
    maxSpeed = 3,
    spriteName = null,
  ) {
    this.x = x;
    this.y = y;
    this.height = height;
    this.width = width;
    this.speed = 0;
    this.acceleration = 0.2;
    this.maxSpeed = maxSpeed;
    this.friction = 0.05;
    this.angle = 0;
    this.damaged = false;
    this.spriteName = spriteName; // e.g. "car_blue"
    if (controlType != "DUMMY") {
      this.sensor = new Sensor(this);
    }
    this.controls = new Controls(controlType);
  }

  #move() {
    // up and down control
    if (this.controls.forward) {
      this.speed += this.acceleration;
    }
    if (this.controls.reverse) {
      this.speed -= this.acceleration;
    }

    if (this.speed > this.maxSpeed) {
      this.speed = this.maxSpeed;
    }
    if (this.speed < -this.maxSpeed / 2) {
      this.speed = -this.maxSpeed / 2;
    }

    // friction
    if (this.speed > 0) {
      this.speed -= this.friction;
    }
    if (this.speed < 0) {
      this.speed += this.friction;
    }
    if (Math.abs(this.speed) < this.friction) {
      this.speed = 0;
    }

    // left and right control
    if (this.speed != 0) {
      const flip = this.speed > 0 ? 1 : -1;
      if (this.controls.left) {
        this.angle += 0.03 * flip;
      }
      if (this.controls.right) {
        this.angle -= 0.03 * flip;
      }
    }

    this.x -= Math.sin(this.angle) * this.speed;
    this.y -= Math.cos(this.angle) * this.speed;
  }

  update(roadBorders, traffic) {
    if (!this.damaged) {
      this.#move();
      this.polygon = this.#createPolygon();
      this.damaged = this.#assessDamage(roadBorders, traffic);
    }

    if (this.sensor) {
      this.sensor.update(roadBorders, traffic);
    }
  }

  #assessDamage(roadBorders, traffic) {
    for (let i = 0; i < roadBorders.length; i++) {
      if (polysIntersect(this.polygon, roadBorders[i])) {
        return true;
      }
    }
    for (let i = 0; i < traffic.length; i++) {
      if (polysIntersect(this.polygon, traffic[i].polygon)) {
        return true;
      }
    }
    return false;
  }

  #createPolygon() {
    const points = [];
    const rad = Math.hypot(this.width, this.height) / 2;
    const alpha = Math.atan2(this.width, this.height);
    points.push({
      x: this.x - Math.sin(this.angle - alpha) * rad,
      y: this.y - Math.cos(this.angle - alpha) * rad,
    });
    points.push({
      x: this.x - Math.sin(this.angle + alpha) * rad,
      y: this.y - Math.cos(this.angle + alpha) * rad,
    });
    points.push({
      x: this.x - Math.sin(Math.PI + this.angle - alpha) * rad,
      y: this.y - Math.cos(Math.PI + this.angle - alpha) * rad,
    });
    points.push({
      x: this.x - Math.sin(Math.PI + this.angle + alpha) * rad,
      y: this.y - Math.cos(Math.PI + this.angle + alpha) * rad,
    });

    return points;
  }

  draw(ctx, color) {
    const img = this.spriteName ? sprites[this.spriteName] : null;

    if (spriteReady(img)) {
      ctx.save();
      ctx.translate(this.x, this.y);
      ctx.rotate(-this.angle + SPRITE_ROTATION);
      if (this.damaged) {
        ctx.filter = "grayscale(1) brightness(0.7)";
      }
      ctx.drawImage(
        img,
        -this.width / 2,
        -this.height / 2,
        this.width,
        this.height,
      );
      ctx.restore();
    } else if (this.polygon) {
      ctx.fillStyle = this.damaged ? "gray" : color;
      ctx.beginPath();
      ctx.moveTo(this.polygon[0].x, this.polygon[0].y);
      for (let i = 1; i < this.polygon.length; i++) {
        ctx.lineTo(this.polygon[i].x, this.polygon[i].y);
      }
      ctx.fill();
    }

    const crash = sprites["crash"];
    if (this.damaged && spriteReady(crash)) {
      const w = this.width * 2.2;
      const h = w * (crash.naturalHeight / crash.naturalWidth);
      ctx.drawImage(crash, this.x - w / 2, this.y - h / 2, w, h);
    }

    if (this.sensor && !this.damaged) {
      this.sensor.draw(ctx);
    }
  }
}
