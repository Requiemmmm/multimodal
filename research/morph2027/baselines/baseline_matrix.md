# MORPH baseline matrix

## Same action space (must run)

- no suffix and frequency-matched random suffix;
- authorized direct image request (positive upper bound);
- nearest-neighbor soft-to-hard projection;
- quantization-aware projection;
- GCG with preamble NLL;
- DeGCG-style BOI margin and BOI CE;
- I-GCG-style multi-coordinate search;
- constrained route-floor joint search;
- best-of-N under a fixed query budget.

## Reference-only or non-comparable

- AutoDAN/PAIR/TAP: text jailbreak search spaces differ;
- UMK/FigStep/HADES: image injection is stronger than the native threat setting;
- Rando et al. Chameleon GCG: refusal flipping, not modality/content control;
- text-to-image attacks: different success definition.

Every comparison must state suffix length, token support, query budget, decoder, model checkpoint, and whether it is native, continuous, bridge, or oracle.
