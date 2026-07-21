// Motion-clip retargeting test: plays authored MotionClips (canonical
// world-space deltas from T-pose) on every available rig and checks the
// resulting WORLD-space hand/head geometry — the same clip must produce the
// same world pose on the capsule, Mixamo GLB, and VRM skeletons despite
// their wildly different local bone frames. Prereqs: same as drive.cjs.
const path = require('path');
const { chromium } = require('playwright-core');

const APP = process.env.APP_URL || 'http://localhost:5173';
const CDP = process.env.CDP_URL || 'http://localhost:9222';
const ART = path.join(__dirname, '.artifacts');

let failures = 0;
function check(name, ok, detail) {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
  if (!ok) failures++;
}

/** Axis-angle → quaternion [x,y,z,w]. */
function quat(axis, angle) {
  const s = Math.sin(angle / 2);
  const n = Math.hypot(...axis);
  return [(axis[0] / n) * s, (axis[1] / n) * s, (axis[2] / n) * s, Math.cos(angle / 2)];
}
const QID = [0, 0, 0, 1];

const ALL_JOINTS = [
  'hips', 'spine', 'chest', 'neck', 'head',
  'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand',
  'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand',
  'leftUpperLeg', 'leftLowerLeg', 'leftFoot',
  'rightUpperLeg', 'rightLowerLeg', 'rightFoot',
];

/** 1 s of pure canonical T-pose (identity deltas), feet free. */
function tposeClip() {
  const frame = ALL_JOINTS.map(() => QID);
  return { fps: 2, joints: ALL_JOINTS, rotations: [frame, frame], pinFeet: false };
}

/**
 * Wave hello: right arm swings up overhead (delta from T-pose is a roll
 * about +Z — T-pose arm points -X, rotating -X toward +Y), forearm wags.
 */
function waveClip() {
  const fps = 30;
  const frames = [];
  for (let i = 0; i < fps * 2; i++) {
    const t = i / fps;
    const raise = Math.min(1, t * 4); // up in 0.25 s, then hold
    const wag = Math.sin(t * 2 * Math.PI * 2.2) * 0.35;
    frames.push([
      quat([0, 0, 1], (-Math.PI / 2 - 0.25) * raise), // rightUpperArm
      quat([0, 0, 1], (-Math.PI / 2) * raise + wag * raise), // rightLowerArm
      quat([1, 0, 0], -0.1 * raise), // head: slight tilt up
    ]);
  }
  return { fps, joints: ['rightUpperArm', 'rightLowerArm', 'head'], rotations: frames };
}

async function samplePose(page) {
  return page.evaluate(() => {
    const j = window.__app.rig.joints;
    const v = j.hips.position.clone();
    const grab = (name) => {
      j[name].getWorldPosition(v);
      return [v.x, v.y, v.z];
    };
    return {
      leftHand: grab('leftHand'),
      rightHand: grab('rightHand'),
      head: grab('head'),
      hips: grab('hips'),
      leftShoulder: grab('leftShoulder'),
      rightShoulder: grab('rightShoulder'),
    };
  });
}

(async () => {
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
  const rigCount = await page
    .waitForFunction(() => window.__app.rigCount >= 3, null, { timeout: 20000 })
    .then(() => page.evaluate(() => window.__app.rigCount))
    .catch(() => page.evaluate(() => window.__app.rigCount ?? 1));
  console.log('rigs available: %d', rigCount);

  for (let r = 0; r < rigCount; r++) {
    const rigName = await page.evaluate(() => window.__app.rig.root.name);
    console.log(`\n=== rig: ${rigName}`);

    // Rest-pose geometry for reference.
    await page.waitForTimeout(300);
    const rest = await samplePose(page);
    const restSep = Math.abs(rest.leftHand[0] - rest.rightHand[0]);

    // --- T-pose identity clip: arms must extend horizontally to ±X ---
    await page.evaluate((clip) => window.__app.playClip(clip), tposeClip());
    await page.waitForTimeout(400);
    const t = await samplePose(page);
    const sep = Math.abs(t.leftHand[0] - t.rightHand[0]);
    const shoulderY = (t.leftShoulder[1] + t.rightShoulder[1]) / 2;
    const armDrop = Math.abs((t.leftHand[1] + t.rightHand[1]) / 2 - shoulderY);
    const torso = t.head[1] - t.hips[1];
    // Rest (arms hanging at ~65°) already separates the hands somewhat, so
    // the discriminating signal is separation relative to torso height:
    // a true T-pose spans well over 1.8× hips→head.
    check('T-pose: hands spread wide', sep > restSep * 1.4 && sep > torso * 1.8,
      `sep=${sep.toFixed(3)} rest=${restSep.toFixed(3)} torso=${torso.toFixed(3)}`);
    check('T-pose: hands at shoulder height', armDrop < torso * 0.35,
      `drop=${armDrop.toFixed(3)}`);
    await page.screenshot({ path: path.join(ART, `clip-tpose-${r}-${rigName}.png`) });
    await page.evaluate(() => window.__app.animator.clipPlayer.stop());

    // --- Wave clip: right hand rises above the head and wags ---
    await page.evaluate((clip) => window.__app.playClip(clip), waveClip());
    await page.waitForTimeout(600); // arm fully raised, mid-wave
    const xs = [];
    let maxY = -Infinity;
    for (let i = 0; i < 10; i++) {
      const p = await samplePose(page);
      xs.push(p.rightHand[0]);
      maxY = Math.max(maxY, p.rightHand[1]);
      await page.waitForTimeout(90);
    }
    const headY = (await samplePose(page)).head[1];
    check('wave: right hand above head', maxY > headY,
      `handY=${maxY.toFixed(3)} headY=${headY.toFixed(3)}`);
    const wagRange = Math.max(...xs) - Math.min(...xs);
    check('wave: hand wags laterally', wagRange > torso * 0.06,
      `range=${wagRange.toFixed(3)}`);
    await page.screenshot({ path: path.join(ART, `clip-wave-${r}-${rigName}.png`) });

    // --- Clip ends → player deactivates, procedural behavior resumes ---
    await page.waitForTimeout(1200);
    const active = await page.evaluate(() => window.__app.animator.clipPlayer.active);
    check('clip auto-finishes', active === false, `active=${active}`);

    if (r < rigCount - 1) {
      await page.click('#btn-character');
      await page.waitForTimeout(600);
    }
  }

  check('zero console errors', consoleErrors.length === 0, consoleErrors.join(' | '));
  console.log(failures ? `\n${failures} FAILURES` : '\nALL PASS');
  await page.close();
  process.exit(failures ? 1 : 0);
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
