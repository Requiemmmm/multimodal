# MORPH data layout

- `raw/`: immutable prompt lists, model manifests, and imported run metadata.
- `processed/`: validated prompt maps, target trajectories, judge outputs, and benchmark manifests.
- `labels/`: human review labels and inter-rater summaries.

Do not place multi-GB model weights in this repository. Keep them under the configured data disk and record the Hub ID, revision, code commit, file presence, and successful load/decode in the run manifest.
