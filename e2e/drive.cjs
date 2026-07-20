// End-to-end smoke test: drives the app in a headless Chromium-family
// browser over CDP. Prerequisites (see .claude/skills/run-app/SKILL.md):
//   1. Vite dev server running on APP_URL (default http://localhost:5173)
//   2. A browser listening on CDP_URL (default http://localhost:9222),
//      launched with --use-fake-device-for-media-stream etc.
//   3. `npm i playwright-core` somewhere require() can find it, and
//      `node e2e/make-wav.cjs` run once to create the test WAV.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright-core');

const APP = process.env.APP_URL || 'http://localhost:5173';
const CDP = process.env.CDP_URL || 'http://localhost:9222';
const ART = path.join(__dirname, '.artifacts');
const WAV = path.join(ART, 'test.wav');

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
  // If the GLB character is available it loads async and becomes the active
  // rig; wait for it so the audio test below exercises the real model.
  const hasModel = await page
    .waitForFunction(() => !document.getElementById('btn-character').hidden, null, { timeout: 15000 })
    .then(() => true)
    .catch(() => false);
  console.log('character model loaded:', hasModel);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(ART, '01-idle.png') });
  console.log('canvas present, __app hook present');

  const sample = () =>
    page.evaluate(() => {
      const { rig, audio } = window.__app;
      const foot = new (rig.root.position.constructor)();
      rig.joints.leftFoot.getWorldPosition(foot);
      return {
        f: { ...audio.features },
        hipsY: rig.joints.hips.position.y,
        armX: rig.joints.leftUpperArm.rotation.x,
        footWorld: { x: foot.x, y: foot.y, z: foot.z },
        audioEl: (() => {
          const el = document.getElementById('audio-el');
          return { paused: el.paused, t: el.currentTime };
        })(),
      };
    });

  console.log('--- 2. Idle baseline');
  const idle = await sample();
  console.log('idle rms=%s hipsY=%s', idle.f.rms.toFixed(4), idle.hipsY.toFixed(4));

  console.log('--- 3. Load WAV through file input');
  await page.setInputFiles('#file-input', WAV);
  await page.waitForTimeout(2500);

  const during = [];
  for (let i = 0; i < 24; i++) {
    during.push(await sample());
    await page.waitForTimeout(125);
  }
  await page.screenshot({ path: path.join(ART, '02-playing.png') });

  const last = during[during.length - 1];
  console.log('audio element: paused=%s currentTime=%s', last.audioEl.paused, last.audioEl.t.toFixed(2));

  const rms = stats(during.map((s) => s.f.rms));
  const bass = stats(during.map((s) => s.f.bass));
  const pulse = stats(during.map((s) => s.f.beatPulse));
  const hips = stats(during.map((s) => s.hipsY));
  const arm = stats(during.map((s) => s.armX));
  const beatsSeen = during.filter((s) => s.f.timeSinceBeat < 0.6).length;
  const lastBpm = last.f.bpm;
  const lastConf = last.f.tempoConfidence;
  const phaseAdvanced = last.f.beatPhase - during[0].f.beatPhase;
  console.log('rms max=%s  bass max=%s  beatPulse max=%s', rms.max.toFixed(4), bass.max.toFixed(4), pulse.max.toFixed(3));
  console.log('samples with a recent beat: %d/24', beatsSeen);
  console.log('tempo: %s BPM  confidence=%s  phase advanced %s beats', lastBpm.toFixed(1), lastConf.toFixed(2), phaseAdvanced.toFixed(1));
  const footDrift = Math.max(
    stats(during.map((s) => s.footWorld.x)).range,
    stats(during.map((s) => s.footWorld.y)).range,
    stats(during.map((s) => s.footWorld.z)).range,
  );
  console.log('hipsY range=%s  armX range=%s  foot drift=%sm', hips.range.toFixed(4), arm.range.toFixed(4), footDrift.toFixed(4));

  console.log('--- 4. Microphone path (fake device)');
  await page.click('#btn-mic');
  await page.waitForTimeout(2000);
  const mic = await sample();
  console.log('mic rms=%s', mic.f.rms.toFixed(4));
  await page.screenshot({ path: path.join(ART, '03-mic.png') });

  let toggleOk = true;
  if (hasModel) {
    console.log('--- 4.5 Character toggle');
    const before = await page.evaluate(() => window.__app.rig.root.name);
    await page.click('#btn-character');
    await page.waitForTimeout(500);
    const after = await page.evaluate(() => window.__app.rig.root.name);
    toggleOk = before !== after;
    console.log('rig before=%s after=%s', before, after);
    await page.screenshot({ path: path.join(ART, '04-toggled.png') });
    await page.click('#btn-character'); // back to the model
  }

  console.log('--- 5. Console errors: %d', consoleErrors.length);
  consoleErrors.slice(0, 10).forEach((e) => console.log('  ERR:', e));

  const failures = [];
  if (last.audioEl.paused || last.audioEl.t <= 0) failures.push('audio element did not play');
  if (rms.max < 0.05) failures.push('rms never rose during playback');
  if (bass.max < 0.1) failures.push('bass energy never registered');
  if (pulse.max < 0.5) failures.push('no beat pulse detected');
  if (beatsSeen < 4) failures.push('too few beats detected');
  if (hips.range < 0.01) failures.push('hips did not bounce');
  // Foot IK: while the hips move, the planted foot should stay put (meters).
  if (footDrift > 0.03) failures.push(`foot slid ${footDrift.toFixed(3)}m despite IK pinning`);
  // The test WAV is authored at exactly 120 BPM.
  if (Math.abs(lastBpm - 120) > 6) failures.push(`tempo estimate off: ${lastBpm.toFixed(1)} BPM (expected ~120)`);
  if (lastConf < 0.6) failures.push(`tempo confidence low: ${lastConf.toFixed(2)}`);
  if (phaseAdvanced < 2) failures.push('beat phase did not advance');
  if (arm.range < 0.05) failures.push('arms did not move');
  if (mic.f.rms < 0.005) failures.push('mic (fake device) produced no signal');
  if (!toggleOk) failures.push('character toggle did not switch rigs');
  if (consoleErrors.length) failures.push(consoleErrors.length + ' console errors');

  console.log(failures.length ? 'RESULT: FAIL — ' + failures.join('; ') : 'RESULT: PASS');
  await browser.close();
  process.exit(failures.length ? 1 : 0);
})().catch((e) => {
  console.error('DRIVER ERROR:', e);
  process.exit(2);
});
