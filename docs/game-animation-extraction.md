# Game animation extraction (reference libraries)

Local-only pipelines used to extract professional humanoid animation from
installed games as reference/base material. **No game assets are committed**
(`public/models/` is gitignored); only bone-name maps and derived overlay
clips live in the repo. Each game's clips convert to the canonical
MotionClip format via `convertNamedSkeletonAnimations` + a per-game bone map
(`FFXV_BONES`, `FF16_BONES`, `BG3_BONES` in `src/animation/library.ts`).

## Key finding

All three AAA titles store CONVERSATIONAL animation as **additive layers over
a base pose** — FFXVI's `dialogue_idle`, BG3's `DIAG_Fidget`, and FFXV's talk
clips all sit near-rest standalone and are meant to be composited. This
directly validates the base+additive architecture (`SchedulePlayer.additive`).

Measured arm velocity (°/s): FFXVI dialogue 1.2, BG3 dialogue 4.7 (subtle
additive); FFXV gesture 29.5, BG3 idle 116 (absolute/big); **our generation
21.7 vs real BEAT2 mocap 22.7** — statistically matched.

## FINAL FANTASY XV (Luminous, `.earc`/`.pka`/`.ani`)
- `ml/`-adjacent `earctool` (dotnet, built against the Flagrum.Core source)
  decrypts EARC archives and splits PKA packs into `.ani`.
- Noesis under Proton converts `.ani` → FBX (`?cmode`, cmd-wrapped; skeleton
  = `<char>.amdl` placed at the variant root so Noesis finds it).
- Bones: `C_Hip/C_Spine1-3/C_Neck1/C_Head`, `L_/R_ Shoulder/UpperArm/Elbow/Hand`.

## FINAL FANTASY XVI (`.pac` + Havok `.anmb`/`.skl`)
- FF16Tools.CLI (Nenkai) under Proton unpacks `0000.pac` (17,493 anim files);
  the character skeleton lives in the nested `chara/c1001/pack/c1001.pac`.
- FF16-Animation-Converter (obilang) — native .NET 9, pure-C# HKLib — converts
  `.anmb` + `body.skl` → glTF (30 fps).
- Bones: `j_hip/j_spine_01-03/j_neck_01/j_head`, `j_clavicle/arm_01/arm_02/hand`.

## Baldur's Gate 3 (Granny `.gr2` via LSLib)
- Stock LSLib doesn't run on Linux (Windows path bug + `granny2.dll` BitKnit).
  Patched to run native .NET 8 — see `docs/lslib-linux.patch` (drops
  LSLibNative, swaps in a managed BitKnit decoder, fixes path validation,
  lenient conform). Build: apply patch to Norbyte/lslib, `dotnet build`.
- `Divine.dll extract-package Models.pak` → animations under
  `_Anims/Humans/_Male/HUM_M_Rig/` (`DIAG_*` = dialogue gestures, `IDLE_*`);
  `convert-model -e conform --conform-path Proxy_HUM_M_FullBody_A.GR2` → GLB
  with skeletal animation.
- Bones: `Root_M/Spine1_M/Chest_M/Neck_M/Head_M`, `Scapula/Shoulder/Elbow/Wrist`.
