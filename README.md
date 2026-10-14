# MORPH New-Server Repository Seed

This bundle is intended to be copied into the root of a fresh MORPH research repository.

## Start here

1. Read `AGENTS.md` and `docs/MORPH_CODEX_NEW_SERVER_GUIDE.md`.
2. Choose a fast local filesystem and set `MORPH_ROOT`, defaulting to `/srv/morph`.
3. Run the Lumina environment bootstrap:

   ```bash
   MORPH_ROOT=/srv/morph bash scripts/bootstrap_lumina_env.sh
   ```

4. Authenticate with Hugging Face and download the primary checkpoint:

   ```bash
   conda activate morph-lumina
   hf auth login
   MORPH_ROOT=/srv/morph bash scripts/download_models.sh lumina
   ```

5. Follow Stage 0 in the detailed guide. Anole's public checkpoint bundles its Chameleon VQ/tokenizer files, so it is the fastest path for the first authorized image smoke test. Lumina additionally needs the external Meta Chameleon VQ files described in the guide.
6. Keep `runs/STATUS.md` updated after each substantive experiment.

## Files

- `AGENTS.md`: concise global behavior and engineering constraints for Codex.
- `docs/MORPH_CODEX_NEW_SERVER_GUIDE.md`: complete background, environment, model downloads, code architecture, MORPH-v0 theory, staged experiments, fallbacks, and global research rules.
- `scripts/bootstrap_lumina_env.sh`: primary environment bootstrap.
- `scripts/download_models.sh`: Hugging Face model downloader.
- `.env.example`: path and runtime variable template.
- `models/manifest.example.yaml`: model provenance template.
- `runs/STATUS.md`: observation/interpretation/next-action log template.

The bootstrap scripts were checked with `bash -n`. Model downloads and GPU execution were not performed in this environment.
