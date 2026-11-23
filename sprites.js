// Sprite loader: images live in ./assets (relative to index.html)
const spriteImages = [
  "car_black.png",
  "car_blue.png",
  "car_red.png",
  "car_yellow.png",
  "crash.png",
];

// If your car sprites point DOWN instead of up, set this to Math.PI
const SPRITE_ROTATION = 0;

// sprites["car_blue"], sprites["crash"], ... (filename without .png)
const sprites = {};
spriteImages.forEach((file) => {
  const img = new Image();
  img.src = "./assets/" + file;
  sprites[file.replace(".png", "")] = img;
});

function spriteReady(img) {
  return img && img.complete && img.naturalWidth > 0;
}
