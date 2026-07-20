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
  const rigCount = await page
    .waitForFunction(() => window.__app.rigCount >= 3, null, { timeout: 20000 })
    .then(() => page.evaluate(() => window.__app.rigCount))
    .catch(() => page.evaluate(() => window.__app.rigCount ?? 1));
  console.log('character model loaded: %s  rigs available: %d', hasModel, rigCount);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(ART, '01-idle.png') });
  console.log('canvas present, __app hook present');

  const sample = () =>
    page.evaluate(() => {
      const { rig, audio } = window.__app;
      const V = rig.root.position.constructor;
      const foot = new V();
      rig.joints.leftFoot.getWorldPosition(foot);
      // Interior elbow angle (PI = straight arm) from world joint positions.
      const sh = new V(), el = new V(), wr = new V();
      rig.joints.leftUpperArm.getWorldPosition(sh);
      rig.joints.leftLowerArm.getWorldPosition(el);
      rig.joints.leftHand.getWorldPosition(wr);
      const elbowAngle = sh.sub(el).angleTo(wr.sub(el));
      return {
        f: { ...audio.features },
        hipsY: rig.joints.hips.position.y,
        armX: rig.joints.leftUpperArm.rotation.x,
        footWorld: { x: foot.x, y: foot.y, z: foot.z },
        move: window.__app.animator?.currentMove ?? -1,
        elbowAngle,
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
  // Offline analysis lands within a second of the file being selected.
  const timeline = await page
    .waitForFunction(() => window.__app.audio.timeline, null, { timeout: 10000 })
    .then(() =>
      page.evaluate(() => {
        const tl = window.__app.audio.timeline;
        return {
          bpm: tl.bpm,
          tempoStrength: tl.tempoStrength,
          beats: tl.beats,
          downbeatOffset: tl.downbeatOffset,
          sections: tl.sections,
          mode: tl.mode,
        };
      }),
    );
  console.log(
    'timeline: mode=%s bpm=%s strength=%s beats=%d sections=%j',
    timeline.mode,
    timeline.bpm.toFixed(1),
    timeline.tempoStrength.toFixed(2),
    timeline.beats.length,
    timeline.sections.map((s) => `${s.level}${s.drop ? '(drop)' : ''}@${s.start.toFixed(1)}`),
  );

  // Beat-grid accuracy vs the authored kick grid (kicks at n*0.5 s).
  const gridErrs = timeline.beats.map((b) => Math.abs(b - Math.round(b / 0.5) * 0.5));
  gridErrs.sort((a, b) => a - b);
  const medianGridErr = gridErrs[Math.floor(gridErrs.length / 2)] ?? 1;
  // Downbeat accuracy: downbeat-offset beats should sit on 2 s bar lines.
  const downbeats = timeline.beats.filter((_, i) => i % 4 === timeline.downbeatOffset % 4);
  const dbErrs = downbeats.map((b) => Math.abs(b - Math.round(b / 2) * 2)).sort((a, b) => a - b);
  const medianDbErr = dbErrs[Math.floor(dbErrs.length / 2)] ?? 1;
  const dropSection = timeline.sections.find((s) => s.drop);
  console.log(
    'beat grid: median err=%sms  downbeat median err=%sms  drop at %s',
    (medianGridErr * 1000).toFixed(1),
    (medianDbErr * 1000).toFixed(1),
    dropSection ? dropSection.start.toFixed(2) + 's' : 'NONE',
  );

  await page.waitForTimeout(2000);
  const during = [];
  for (let i = 0; i < 40; i++) {
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
  const beatsSeen = during.filter((s) => s.f.timeSinceBeat < 0.6 || s.f.beatPulse > 0.05).length;
  const sectionRise = during[during.length - 1].f.section - during[0].f.section;
  console.log('section feature: first=%s last=%s (drop crossed)', during[0].f.section.toFixed(2), during[during.length - 1].f.section.toFixed(2));
  const lastBpm = last.f.bpm;
  const lastConf = last.f.tempoConfidence;
  const phaseAdvanced = last.f.beatPhase - during[0].f.beatPhase;
  console.log('rms max=%s  bass max=%s  beatPulse max=%s', rms.max.toFixed(4), bass.max.toFixed(4), pulse.max.toFixed(3));
  console.log('samples with a recent beat: %d/40', beatsSeen);
  console.log('tempo: %s BPM  confidence=%s  phase advanced %s beats', lastBpm.toFixed(1), lastConf.toFixed(2), phaseAdvanced.toFixed(1));
  const footDrift = Math.max(
    stats(during.map((s) => s.footWorld.x)).range,
    stats(during.map((s) => s.footWorld.y)).range,
    stats(during.map((s) => s.footWorld.z)).range,
  );
  const movesSeen = [...new Set(during.map((s) => s.move))];
  const minElbow = Math.min(...during.map((s) => s.elbowAngle));
  console.log('hipsY range=%s  armX range=%s  foot drift=%sm', hips.range.toFixed(4), arm.range.toFixed(4), footDrift.toFixed(4));
  console.log('arm moves seen: %j  min elbow angle=%s rad', movesSeen, minElbow.toFixed(2));

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

  console.log('--- 4.7 Speech classification + facial animation');
  // Switch to the VRM avatar (the rig with a face) before speech playback.
  for (let i = 0; i < 4; i++) {
    const hasFace = await page.evaluate(() => !!window.__app.rig.face);
    if (hasFace) break;
    await page.click('#btn-character');
    await page.waitForTimeout(200);
  }
  const onFaceRig = await page.evaluate(() => !!window.__app.rig.face);
  await page.setInputFiles('#file-input', path.join(ART, 'speech.wav'));
  const speechMode = await page
    .waitForFunction(
      () => window.__app.audio.timeline && window.__app.audio.timeline.duration < 10,
      null,
      { timeout: 10000 },
    )
    .then(() => page.evaluate(() => window.__app.audio.timeline.mode));
  console.log('speech.wav classified as: %s  on face-capable rig: %s', speechMode, onFaceRig);

  // Sample the face for ~3 s of speech playback.
  let faceStats = { maxMouth: 0, blinkCount: 0, mood: 0 };
  if (onFaceRig) {
    for (let i = 0; i < 30; i++) {
      const d = await page.evaluate(() => window.__app.animator.faceAnimator?.debug() ?? null);
      if (d) {
        faceStats.maxMouth = Math.max(faceStats.maxMouth, d.aa + d.ih + d.ou);
        faceStats.blinkCount = d.blinkCount;
        faceStats.mood = d.mood;
      }
      await page.waitForTimeout(100);
    }
    console.log('face: max mouth=%s blinks=%d mood=%s', faceStats.maxMouth.toFixed(2), faceStats.blinkCount, faceStats.mood.toFixed(2));
    await page.screenshot({ path: path.join(ART, '05-vrm-speech.png') });
  }

  console.log('--- 5. Console errors: %d', consoleErrors.length);
  consoleErrors.slice(0, 10).forEach((e) => console.log('  ERR:', e));

  const failures = [];
  if (last.audioEl.paused || last.audioEl.t <= 0) failures.push('audio element did not play');
  if (rms.max < 0.05) failures.push('rms never rose during playback');
  if (bass.max < 0.1) failures.push('bass energy never registered');
  if (pulse.max < 0.5) failures.push('no beat pulse detected');
  if (beatsSeen < 8) failures.push('too few beats detected');
  // Offline timeline accuracy (test.wav ground truth).
  if (timeline.mode !== 'music') failures.push(`test.wav classified as ${timeline.mode}, expected music`);
  if (Math.abs(timeline.bpm - 120) > 3) failures.push(`timeline BPM ${timeline.bpm.toFixed(1)}, expected ~120`);
  if (medianGridErr > 0.04) failures.push(`beat grid median error ${(medianGridErr * 1000).toFixed(0)}ms > 40ms`);
  if (medianDbErr > 0.06) failures.push(`downbeat median error ${(medianDbErr * 1000).toFixed(0)}ms > 60ms`);
  if (!dropSection) failures.push('no drop section detected');
  else if (Math.abs(dropSection.start - 6) > 1.2) failures.push(`drop at ${dropSection.start.toFixed(1)}s, expected ~6s`);
  if (sectionRise < 0.25) failures.push(`section feature did not rise across the drop (${sectionRise.toFixed(2)})`);
  if (hips.range < 0.01) failures.push('hips did not bounce');
  // Foot IK: while the hips move, the planted foot should stay put (meters).
  if (footDrift > 0.03) failures.push(`foot slid ${footDrift.toFixed(3)}m despite IK pinning`);
  // Choreography should rotate through at least one phrase switch in-window.
  if (movesSeen.length < 2) failures.push(`arm move never changed (saw ${JSON.stringify(movesSeen)})`);
  // Elbows must visibly bend at some point (straight arm = PI ≈ 3.14 rad).
  if (minElbow > 2.6) failures.push(`elbow never bent (min interior angle ${minElbow.toFixed(2)} rad)`);
  // The test WAV is authored at exactly 120 BPM.
  if (Math.abs(lastBpm - 120) > 6) failures.push(`tempo estimate off: ${lastBpm.toFixed(1)} BPM (expected ~120)`);
  if (lastConf < 0.6) failures.push(`tempo confidence low: ${lastConf.toFixed(2)}`);
  if (phaseAdvanced < 2) failures.push('beat phase did not advance');
  if (arm.range < 0.05) failures.push('arms did not move');
  if (mic.f.rms < 0.005) failures.push('mic (fake device) produced no signal');
  if (!toggleOk) failures.push('character toggle did not switch rigs');
  if (speechMode !== 'speech') failures.push(`speech.wav classified as ${speechMode}, expected speech`);
  if (rigCount >= 3) {
    if (!onFaceRig) failures.push('could not switch to a face-capable rig');
    else {
      if (faceStats.maxMouth < 0.1) failures.push(`lip sync silent during speech (max mouth ${faceStats.maxMouth.toFixed(2)})`);
      if (faceStats.blinkCount < 1) failures.push('never blinked during 3s of speech');
    }
  }
  if (consoleErrors.length) failures.push(consoleErrors.length + ' console errors');

  console.log(failures.length ? 'RESULT: FAIL — ' + failures.join('; ') : 'RESULT: PASS');
  await browser.close();
  process.exit(failures.length ? 1 : 0);
})().catch((e) => {
  console.error('DRIVER ERROR:', e);
  process.exit(2);
});
