// Faces on a Visium-like grid behind the navbar, each looking at its
// neighbours in turn, like the logo
(() => {
  const navbar = document.querySelector(".navbar");
  if (!navbar) return;
  const canvas = document.createElement("canvas");
  canvas.className = "spots-bg";
  canvas.setAttribute("aria-hidden", "true");
  navbar.prepend(canvas);
  const ctx = canvas.getContext("2d");
  const colors = ["#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"];
  const d = 26;
  const r = 10;
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let spots = [];

  function build() {
    const w = navbar.clientWidth;
    const h = navbar.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Each spot takes the label of its nearest seed, giving tissue-like domains
    const seeds = Array.from({ length: Math.ceil(w / 150) + 1 }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      c: colors[Math.floor(Math.random() * colors.length)],
    }));
    spots = [];
    const dy = (d * Math.sqrt(3)) / 2;
    for (let j = 0; j * dy < h + r; j++) {
      for (let x = ((j % 2) * d) / 2; x < w + r; x += d) {
        if (Math.random() < 0.1) continue;
        const y = j * dy;
        const seed = seeds.reduce((a, b) =>
          Math.hypot(a.x - x, a.y - y) < Math.hypot(b.x - x, b.y - y) ? a : b);
        const a = Math.random() * 2 * Math.PI;
        spots.push({ x, y, c: seed.c, from: a, to: a, t0: 0, next: 0 });
      }
    }
    for (const s of spots) {
      s.nbrs = spots.filter(
        (o) => o !== s && Math.hypot(o.x - s.x, o.y - s.y) < d * 1.01);
    }
  }

  function gaze(s, now) {
    if (now >= s.next) {
      s.from = gaze(Object.assign({}, s, { next: Infinity }), now);
      const o = s.nbrs[Math.floor(Math.random() * s.nbrs.length)];
      s.to = o ? Math.atan2(o.y - s.y, o.x - s.x) : Math.random() * 2 * Math.PI;
      s.t0 = now;
      s.next = now + 2000 + Math.random() * 5000;
    }
    let u = Math.min((now - s.t0) / 300, 1);
    u = u * u * (3 - 2 * u);
    const turn = Math.atan2(Math.sin(s.to - s.from), Math.cos(s.to - s.from));
    return s.from + u * turn;
  }

  function disc(x, y, rad, fill) {
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.arc(x, y, rad, 0, 2 * Math.PI);
    ctx.fill();
  }

  function face(s, a) {
    const gx = Math.cos(a);
    const gy = Math.sin(a);
    const cx = s.x + 0.12 * r * gx;
    const cy = s.y + 0.12 * r * gy;
    disc(s.x, s.y, r, s.c);
    for (const side of [-1, 1]) {
      const ex = cx + side * 0.32 * r;
      const ey = cy - 0.15 * r;
      disc(ex, ey, 0.26 * r, "#fff");
      disc(ex + 0.12 * r * gx, ey + 0.12 * r * gy, 0.13 * r, "#1a1a1a");
    }
    ctx.strokeStyle = "#1a1a1a";
    ctx.lineWidth = 0.08 * r;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.arc(cx, cy + 0.05 * r, 0.38 * r, 0.2 * Math.PI, 0.8 * Math.PI);
    ctx.stroke();
  }

  function frame(now) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const s of spots) face(s, gaze(s, now));
    if (!still) requestAnimationFrame(frame);
  }

  // Rebuild when the navbar changes size, e.g. the mobile menu opening
  new ResizeObserver(() => {
    build();
    if (still) frame(0);
  }).observe(navbar);
  if (!still) requestAnimationFrame(frame);
})();
