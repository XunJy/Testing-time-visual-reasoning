# Single-image differential-prompt pilot

Status: prompt override frozen; GPU run pending.

This is a post-hoc diagnostic derived from the frozen OpenAI CLIP full run. It
does not modify that run and is not evidence of dataset-level generalisation.

## Selected sample

- CUB image id: `2738`
- Full-run sample index: `1314`
- Relative path:
  `048.European_Goldfinch/European_Goldfinch_0047_33332.jpg`
- Baseline Top-1: `American Goldfinch` (class 46; incorrect)
- Ground truth: `European Goldfinch` (class 47; baseline rank 2)
- Official FuDD Top-1: `American Goldfinch` (class 46; still incorrect)
- Official FuDD ground-truth rank: 2

The image appears to show an immature bird. Its plain buff-brown head lacks an
adult European Goldfinch's red face mask, while the broad saturated yellow
patch across the black wing remains clearly visible. Cornell describes this
juvenile pattern and the retained yellow wing patch, and BTO explicitly notes
that newly fledged juveniles lack red facial plumage. The official class-pair
prompts therefore contain an age-specific mismatch for this sample.

## Staged intervention

Stage 1 replaces only the four descriptions for class pair `(46, 47)` and
keeps the other 44 class pairs from the image's original Top-10 unchanged.
This preserves prompt count and isolates the critical baseline confusion. If
that is insufficient, stage 2 can expand to the nine pairs involving class 47;
only then would a full 45-pair rewrite be considered.

The replacement was designed directly by Codex after seeing the image, the
target label, and the frozen predictions. It is intentionally label-aware and
post-hoc; it is not a blind intervention and cannot support a generalisation
claim. `generation_prompt.md` is retained only as an unexecuted earlier draft.

The exact four-pair intervention is frozen in `pair_46_47_override.json`. The
runner validates it, encodes the selected image once, recomputes the 200-class
baseline, then applies official and overridden FuDD to the same baseline
Top-10 with no score fusion.

## Identification sources

- [Cornell: American Goldfinch identification](https://www.allaboutbirds.org/guide/American_Goldfinch/id/)
- [Cornell: European Goldfinch identification](https://www.allaboutbirds.org/guide/European_Goldfinch/id/)
- [BTO: Goldfinch](https://www.bto.org/learn/about-birds/birdfacts/goldfinch)
- [RSPB: Goldfinch](https://www.rspb.org.uk/birds-and-wildlife/goldfinch)
