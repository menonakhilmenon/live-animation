---
name: run-app
description: Launch the Vite dev server and drive the audio-driven humanoid app in a headless browser (Brave flatpak via CDP), including the e2e smoke test with synthetic beat audio.
---

# Run and test the app

Verified on this machine (Fedora ostree, no system Chromium; Brave installed
as flatpak `com.brave.Browser`; no `chromium-cli`).

## 1. Dev server

```bash
npm run dev -- --port 5173 --strictPort >/tmp/vite.log 2>&1 &
timeout 30 bash -c 'until curl -sf http://localhost:5173 >/dev/null; do sleep 1; done'
```

Stop it with `lsof -ti:5173 -sTCP:LISTEN | xargs -r kill`.

## 2. Headless browser with CDP

Brave (Chromium-based) works headless from the flatpak. The fake-media flags
make `getUserMedia` succeed without a prompt (the fake mic emits a quiet tone,
so mic-driven features register a small but nonzero RMS):

```bash
flatpak run com.brave.Browser --headless=new --remote-debugging-port=9222 \
  --no-first-run --no-default-browser-check --disable-gpu \
  --autoplay-policy=no-user-gesture-required \
  --use-fake-ui-for-media-stream --use-fake-device-for-media-stream \
  about:blank >/tmp/brave.log 2>&1 &
timeout 40 bash -c 'until curl -sf http://localhost:9222/json/version >/dev/null; do sleep 1; done'
```

Stop it with `lsof -ti:9222 -sTCP:LISTEN | xargs -r kill`.

## 3. E2E smoke test

`playwright-core` (driver only, no browser download) must be resolvable —
install it in a scratch dir and run from there, or `npm i -D playwright-core`.

```bash
npm run fetch:model       # once: downloads public/models/Xbot.glb (else the
                          # model steps are skipped and the capsule rig is used)
node e2e/make-wav.cjs     # writes e2e/.artifacts/test.wav (120 BPM kick pattern)
node e2e/drive.cjs        # exits 0 on PASS; env: APP_URL, CDP_URL
```

The driver loads the page, feeds the WAV through the file input, samples
`window.__app` (dev-only hook in `src/main.ts` exposing `{ rig, audio }`),
and asserts: audio plays, RMS/bass rise, beats are detected, hips bounce,
arms move, mic path yields signal, zero console errors. Screenshots land in
`e2e/.artifacts/` — look at them; a blank canvas means WebGL failed.

## Gotchas actually hit

- TS 5.7+ needs `Uint8Array<ArrayBuffer>` for analyser buffers.
- The browser's automatic `/favicon.ico` fetch does not appear in
  Playwright `response` events but does log a console 404 — index.html
  ships a data-URI icon so the zero-console-errors assertion holds.
- `createMediaElementSource` can only be called once per `<audio>` element
  (`src/ui.ts` guards this).
