# lemonmatrix.github.io

Results data and public leaderboard for [LemonMatrix](https://github.com/Bekhouche/LemonMatrix),
a benchmarking control-plane for AMD's Lemonade runtime. This repo holds no
code beyond the site generator -- the CLI/library lives in the code repo.

Live site: https://lemonmatrix.github.io

## Layout

- `results/<profile>/*.json` -- LLM and router run results.
- `results/<profile>/<modality>/*.json` -- non-text modality results
  (`classify`, `tts`, `stt`, `imagegen`, `audiogen`, `embeddings`, `rerank`,
  `meshgen`).
- `results/<profile>/trials/*.json` -- raw per-trial sidecars behind each
  result above.
- `scripts/build_site.py` -- reads everything under `results/` and renders
  the single self-contained `index.html` the site serves. No server, no
  build framework: plain static HTML with the data embedded inline and
  filtering/sorting done client-side.

## Submitting a result

Open a PR adding a result JSON under `results/<profile>/`. CI
(`.github/workflows/validate-results.yml`) validates it against LemonMatrix's
schemas before merge -- see the code repo's `README.md` for how to generate
one with `lemonmatrix run` / `lemonmatrix sweep` against your own local
Lemonade instance. Every number on the leaderboard comes from a
contributor's own machine talking to its own Lemonade instance; nothing here
is measured by this repo or by any hosted backend.

## Regenerating the site

```
python3 scripts/build_site.py
```

`.github/workflows/deploy-pages.yml` does this automatically on every push
to `main` and publishes the result via GitHub Pages (requires this repo's
Settings > Pages > Source set to "GitHub Actions").
