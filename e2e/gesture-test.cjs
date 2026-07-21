// Plays an EMAGE-generated MotionClip (e2e/.artifacts/gesture-clip.json,
// produced by ml/generate.py) on the VRM rig and sanity-checks the result:
// the character must stay upright (catches wrong up-axis conventions),
// keep its hips near rest, and actually gesture with its hands.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const APP = process.env.APP_URL || 'http://localhost:5173';
const CDP = process.env.CDP_URL || 'http://localhost:9222';
const ART = path.join(__dirname, '.artifacts');
const CLIP = path.join(ART, 'gesture-clip.json');

let failures = 0;
function check(name, ok, detail) {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
  if (!ok) failures++;
}

(async () => {
  if (!fs.existsSync(CLIP)) {
    console.error('Missing %s — run ml/generate.py first.', CLIP);
    process.exit(2);
  }
  const clip = JSON.parse(fs.readFileSync(CLIP, 'utf8'));
  console.log(
    'clip: %d joints × %d frames @ %d fps (%ss)',
    clip.joints.length, clip.rotations.length, clip.fps,
    (clip.rotations.length / clip.fps).toFixed(1),
  );
  const flat = clip.rotations.flat(2).concat(clip.hipsPosition?.flat() ?? []);
  check('clip has no NaN/undefined', flat.every((v) => Number.isFinite(v)));

  const browser = await chromium.connectOverCDP(CDP);
  const ctx = browser.contexts()[0] ?? (await browser.newContext());
  const page = await ctx.newPage();
  const consoleErrors = [];
  page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()));
  page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + e.message));

  await page.goto(APP, { waitUntil: 'load', timeout: 30000 });
  await page.waitForSelector('#app canvas', { timeout: 20000 });
  await page.waitForFunction(() => window.__app?.rig && window.__app?.animator, null, {
    timeout: 20000,
  });
  await page
    .waitForFunction(() => window.__app.rigCount >= 3, null, { timeout: 20000 })
    .catch(() => {});
  const rigName = await page.evaluate(() => window.__app.rig.root.name);
  console.log('rig: %s', rigName);

  const sample = () =>
    page.evaluate(() => {
      const j = window.__app.rig.joints;
      const v = j.hips.position.clone();
      const grab = (n) => {
        j[n].getWorldPosition(v);
        return [v.x, v.y, v.z];
      };
      return {
        lh: grab('leftHand'), rh: grab('rightHand'),
        head: grab('head'), hips: grab('hips'),
        lf: grab('leftFoot'), rf: grab('rightFoot'),
      };
    });

  const rest = await sample();
  await page.evaluate((c) => window.__app.playClip(c), clip);

  let handTravel = 0;
  let minHeadAboveHips = Infinity;
  let maxHipsDrift = 0;
  let prev = null;
  const seconds = Math.min(6, clip.rotations.length / clip.fps);
  const steps = Math.floor(seconds * 8);
  for (let i = 0; i < steps; i++) {
    const p = await sample();
    if (prev) {
      handTravel += Math.hypot(...p.rh.map((v, k) => v - prev.rh[k]));
      handTravel += Math.hypot(...p.lh.map((v, k) => v - prev.lh[k]));
    }
    minHeadAboveHips = Math.min(minHeadAboveHips, p.head[1] - p.hips[1]);
    maxHipsDrift = Math.max(
      maxHipsDrift,
      Math.hypot(p.hips[0] - rest.hips[0], p.hips[2] - rest.hips[2]),
    );
    if (i === Math.floor(steps / 2)) {
      await page.screenshot({ path: path.join(ART, 'gesture-mid.png') });
    }
    prev = p;
    await page.waitForTimeout(125);
  }
  await page.screenshot({ path: path.join(ART, 'gesture-late.png') });

  const restTorso = rest.head[1] - rest.hips[1];
  check('character stays upright', minHeadAboveHips > restTorso * 0.7,
    `minHeadAboveHips=${minHeadAboveHips.toFixed(3)} restTorso=${restTorso.toFixed(3)}`);
  check('hands actually gesture', handTravel > 0.5,
    `total hand travel=${handTravel.toFixed(2)}m over ${seconds.toFixed(1)}s`);
  check('hips stay grounded', maxHipsDrift < 0.5, `driftXZ=${maxHipsDrift.toFixed(3)}m`);
  const active = await page.evaluate(() => window.__app.animator.clipPlayer.active);
  check('clip still playing (or finished cleanly)', active || seconds >= clip.rotations.length / clip.fps);
  check('zero console errors', consoleErrors.length === 0, consoleErrors.join(' | '));

  console.log(failures ? `\n${failures} FAILURES` : '\nALL PASS');
  await page.close();
  process.exit(failures ? 1 : 0);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
