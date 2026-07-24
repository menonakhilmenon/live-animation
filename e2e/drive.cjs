// End-to-end smoke test: drives the app in a headless Chromium-family
// browser over CDP and exercises the reactive speech path — load a
// speech-like WAV, confirm it classifies as speech, and check that the
// character lip-syncs, gestures with its hands, keeps its hips near rest
// (not dancing), and keeps its planted foot pinned by IK. Prerequisites
// (see .claude/skills/run-app/SKILL.md):
//   1. Vite dev server running on APP_URL (default http://localhost:5173)
//   2. A browser listening on CDP_URL (default http://localhost:9222),
//      launched with --use-fake-device-for-media-stream etc.
//   3. `npm i playwright-core` somewhere require() can find it, and
//      `node e2e/make-wav.cjs` run once to create speech.wav.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const APP = process.env.APP_URL || 'http://localhost:5173';
const CDP = process.env.CDP_URL || 'http://localhost:9222';
const ART = path.join(__dirname, '.artifacts');
const WAV = path.join(ART, 'speech.wav');

function stats(arr) {
  const min = Math.min(...arr), max = Math.max(...arr);
  return { min, max, range: max - min };
}

(async () => {
  fs.mkdirSync(ART, { recursive: true });
  if (!fs.existsSync(WAV)) {
    console.error('Missing %s — run `node e2e/make-wav.cjs` first.', WAV);
    process.exit(2);
  }

  const browser = await chromium.connectOverCDP(CDP);
  const ctx = browser.contexts()[0] ?? (await browser.newContext());
  const page = await ctx.newPage();

  const consoleErrors = [];
  page.on('console', (m) => m.type() === 'error' && consoleErrors.push(m.text()));
  page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + e.message));

  console.log('--- 1. Load page');
  await page.goto(APP, { waitUntil: 'load', timeout: 30000 });
  await page.waitForSelector('#app canvas', { timeout: 20000 });
  await page.waitForFunction(() => window.__app?.rig && window.__app?.audio, null, { timeout: 10000 });
  const rigCount = await page
    .waitForFunction(() => window.__app.rigCount >= 3, null, { timeout: 20000 })
    .then(() => page.evaluate(() => window.__app.rigCount))
    .catch(() => page.evaluate(() => window.__app.rigCount ?? 1));
  console.log('rigs available: %d', rigCount);
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(ART, '01-idle.png') });

  console.log('--- 2. Switch to a face-capable rig');
  for (let i = 0; i < 4; i++) {
    const hasFace = await page.evaluate(() => !!window.__app.rig.face);
    if (hasFace) break;
    await page.click('#btn-character');
    await page.waitForTimeout(200);
  }
  const onFaceRig = await page.evaluate(() => !!window.__app.rig.face);
  console.log('on face-capable rig: %s', onFaceRig);

  console.log('--- 3. Load speech.wav');
  await page.setInputFiles('#file-input', WAV);
  const speechMode = await page
    .waitForFunction(() => window.__app.audio.timeline, null, { timeout: 10000 })
    .then(() => page.evaluate(() => window.__app.audio.timeline.mode));
  console.log('speech.wav classified as: %s', speechMode);

  const sample = () =>
    page.evaluate(() => {
      const { rig, animator } = window.__app;
      const V = rig.root.position.constructor;
      const foot = new V(), hl = new V(), hr = new V();
      rig.joints.leftFoot.getWorldPosition(foot);
      rig.joints.leftHand.getWorldPosition(hl);
      rig.joints.rightHand.getWorldPosition(hr);
      return {
        f: { ...window.__app.audio.features },
        face: animator.faceAnimator?.debug() ?? null,
        hipsY: rig.joints.hips.position.y,
        footWorld: { x: foot.x, y: foot.y, z: foot.z },
        handL: { x: hl.x, y: hl.y, z: hl.z },
        handR: { x: hr.x, y: hr.y, z: hr.z },
        audioEl: (() => {
          const el = document.getElementById('audio-el');
          return { paused: el.paused, t: el.currentTime };
        })(),
      };
    });

  await page.waitForTimeout(1000);
  const during = [];
  let faceStats = { maxMouth: 0, blinkCount: 0 };
  for (let i = 0; i < 30; i++) {
    const s = await sample();
    during.push(s);
    if (s.face) {
      faceStats.maxMouth = Math.max(faceStats.maxMouth, s.face.aa + s.face.ih + s.face.ou);
      faceStats.blinkCount = s.face.blinkCount;
    }
    await page.waitForTimeout(100);
  }
  await page.screenshot({ path: path.join(ART, '02-speech.png') });

  const last = during[during.length - 1];
  const rms = stats(during.map((s) => s.f.rms));
  const hips = stats(during.map((s) => s.hipsY));
  const footDrift = Math.max(
    stats(during.map((s) => s.footWorld.x)).range,
    stats(during.map((s) => s.footWorld.y)).range,
    stats(during.map((s) => s.footWorld.z)).range,
  );
  const handRange = (key) =>
    Math.max(...['x', 'y', 'z'].map((ax) => stats(during.map((s) => s[key][ax])).range));
  const handTravel = Math.max(handRange('handL'), handRange('handR'));
  console.log('audio element: paused=%s currentTime=%s', last.audioEl.paused, last.audioEl.t.toFixed(2));
  console.log('rms max=%s  hips range=%s  foot drift=%sm  hand travel=%sm',
    rms.max.toFixed(4), hips.range.toFixed(4), footDrift.toFixed(4), handTravel.toFixed(3));
  console.log('face: max mouth=%s blinks=%d', faceStats.maxMouth.toFixed(2), faceStats.blinkCount);

  console.log('--- 4. Console errors: %d', consoleErrors.length);
  consoleErrors.slice(0, 10).forEach((e) => console.log('  ERR:', e));

  const failures = [];
  if (last.audioEl.paused || last.audioEl.t <= 0) failures.push('audio element did not play');
  if (rms.max < 0.02) failures.push('rms never rose during playback');
  if (speechMode !== 'speech') failures.push(`speech.wav classified as ${speechMode}, expected speech`);
  if (footDrift > 0.03) failures.push(`foot slid ${footDrift.toFixed(3)}m despite IK pinning`);
  if (hips.range > 0.02) failures.push(`hips moved too much for speech (range ${hips.range.toFixed(3)})`);
  if (onFaceRig) {
    if (faceStats.maxMouth < 0.1) failures.push(`lip sync silent during speech (max mouth ${faceStats.maxMouth.toFixed(2)})`);
    if (handTravel < 0.03) failures.push(`no visible speech gestures (hand travel ${handTravel.toFixed(3)}m)`);
  }
  if (consoleErrors.length) failures.push(consoleErrors.length + ' console errors');

  console.log(failures.length ? 'RESULT: FAIL — ' + failures.join('; ') : 'RESULT: PASS');
  await browser.close();
  process.exit(failures.length ? 1 : 0);
})().catch((e) => {
  console.error('DRIVER ERROR:', e);
  process.exit(2);
});
