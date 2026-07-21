// Full "dynamic animation suite" loop through the real UI: type text, pick
// an emotion, click Speak — the sidecar (ml/server.py, must be running on
// :8600) synthesizes speech + generates gestures, and the page plays both.
// Asserts gesturing hands, audio-driven lip sync, and the emotion's mood
// bias reaching the face.
const path = require('path');
const http = require('http');
const { chromium } = require('playwright-core');

const APP = process.env.APP_URL || 'http://localhost:5173';
const CDP = process.env.CDP_URL || 'http://localhost:9222';
const ART = path.join(__dirname, '.artifacts');

let failures = 0;
function check(name, ok, detail) {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
  if (!ok) failures++;
}

function sidecarUp() {
  return new Promise((resolve) => {
    http.get('http://127.0.0.1:8600/health', (r) => resolve(r.statusCode === 200))
      .on('error', () => resolve(false));
  });
}

(async () => {
  if (!(await sidecarUp())) {
    console.error('Sidecar not running — start it: ml/.venv/bin/python ml/server.py');
    process.exit(2);
  }

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
  console.log('rig: %s', await page.evaluate(() => window.__app.rig.root.name));

  await page.fill('#speak-text', 'Hello! I am so glad you are here. Let me show you what I can do with my hands while I talk.');
  await page.selectOption('#speak-emotion', 'excited');
  await page.click('#btn-speak');

  // Generation takes a few seconds; playback starts when audio plays.
  await page.waitForFunction(
    () => window.__app.animator.schedulePlayer.active || window.__app.animator.clipPlayer.active,
    null,
    { timeout: 60000 },
  );
  const mode = await page.evaluate(() =>
    window.__app.animator.schedulePlayer.active ? 'schedule' : 'raw-clip',
  );
  console.log('playing via:', mode);
  check('prebaked schedule mode active', mode === 'schedule');

  const sample = () =>
    page.evaluate(() => {
      const j = window.__app.rig.joints;
      const v = j.hips.position.clone();
      const grab = (n) => {
        j[n].getWorldPosition(v);
        return [v.x, v.y, v.z];
      };
      const face = window.__app.animator.faceAnimator?.debug() ?? {};
      const audioEl = document.getElementById('audio-el');
      return { rh: grab('rightHand'), lh: grab('leftHand'), face, t: audioEl.currentTime };
    });

  let handTravel = 0;
  let maxMouth = 0;
  let maxMood = -Infinity;
  let maxJump = 0;
  let audioAdvanced = false;
  let prev = null;
  for (let i = 0; i < 32; i++) {
    const p = await sample();
    if (prev) {
      const jump = Math.hypot(...p.rh.map((v, k) => v - prev.rh[k]));
      maxJump = Math.max(maxJump, jump);
      handTravel += jump;
      handTravel += Math.hypot(...p.lh.map((v, k) => v - prev.lh[k]));
      if (p.t > prev.t) audioAdvanced = true;
    }
    maxMouth = Math.max(maxMouth, (p.face.aa ?? 0) + (p.face.ih ?? 0) + (p.face.ou ?? 0));
    maxMood = Math.max(maxMood, p.face.mood ?? -Infinity);
    if (i === 16) await page.screenshot({ path: path.join(ART, 'speak-mid.png') });
    prev = p;
    await page.waitForTimeout(125);
  }

  check('audio plays', audioAdvanced);
  const hasTrack = await page.evaluate(() => !!window.__app.animator.faceAnimator?.hasVisemeTrack);
  check('phoneme viseme track installed', hasTrack);
  check('hands gesture during speech', handTravel > 0.4, `travel=${handTravel.toFixed(2)}m`);
  // Artifact guard: hand may not teleport between 125 ms samples. Fast
  // legitimate gesticulation peaks ~2.5 m/s (~0.31 m/sample); a pose pop
  // is an order of magnitude beyond that.
  check('motion is continuous (no pose pops)', maxJump < 0.45, `maxJump=${maxJump.toFixed(3)}m`);
  check('lip sync moves the mouth', maxMouth > 0.1, `maxMouth=${maxMouth.toFixed(2)}`);
  check('excited mood reaches the face', maxMood > 0.4, `maxMood=${maxMood.toFixed(2)}`);
  check('zero console errors', consoleErrors.length === 0, consoleErrors.join(' | '));

  console.log(failures ? `\n${failures} FAILURES` : '\nALL PASS');
  await page.close();
  process.exit(failures ? 1 : 0);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
