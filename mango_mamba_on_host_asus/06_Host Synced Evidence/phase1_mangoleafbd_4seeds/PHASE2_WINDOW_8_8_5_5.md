# Phase 2 — aligned 320px windows

Purpose: improve the current 320px fixed-window inter-Mamba result without changing the trainer or model source.

Only change from the 320px baseline: `--window-size 8 8 5 5`.

At 320px, feature maps are `80 -> 40 -> 20 -> 10`; these window sizes divide every map exactly. The current `[7, 7, 7, 7]` pads those maps.

Run seeds: `123`, `42` first. Output on ASUS is restricted to `results/test new mangomamba baseline to surpase 99.75 acc/phase2_window_8_8_5_5/`.
