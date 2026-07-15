# Unexecuted historical draft: prompt for an inexpensive vision model

This draft was not used. The user chose to have Codex design and evaluate the
replacement descriptions directly. The completed intervention is therefore
label-aware and not blind; see `pair_46_47_override.json` and `README.md`.

Attach the selected bird image, then send the text below verbatim.

```text
You are a fine-grained bird-visual-difference writer for a CLIP reranking
experiment. An image is attached, but this is a blind test: you are NOT given
the correct class. Do not guess, state, or rank the image's class.

We need to replace one FuDD class-pair entry:

- class 46: American Goldfinch
- class 47: European Goldfinch

The current descriptions failed on this image:

1. head pattern
   - American: a photograph of an american goldfinch, a type of bird, with a
     black cap on its head.
   - European: a photograph of a european goldfinch, a type of bird, with a
     red face mask.
2. wing bars
   - American: a photograph of an american goldfinch, a type of bird, with
     white wing bars.
   - European: a photograph of a european goldfinch, a type of bird, with
     yellow wing bars.
3. body color
   - American: a photograph of an american goldfinch, a type of bird, with
     bright yellow body color.
   - European: a photograph of a european goldfinch, a type of bird, with
     muted yellow body color.
4. bill shape
   - American: a photograph of an american goldfinch, a type of bird, with a
     conical bill shape.
   - European: a photograph of a european goldfinch, a type of bird, with a
     pointed bill shape.

Generate exactly FOUR replacement attribute pairs. Use the attached image only
to prioritise features that are actually visible at this scale; remain
symmetric and factually describe BOTH species. Account for juvenile, female,
and non-breeding plumages, so an adult-male-only hallmark must not be the sole
evidence. Prefer concrete local cues such as wing-patch colour and geometry,
tail markings, bill proportions, and the spatial relationship of markings.

Rules:

- Each pair must compare the same visual attribute on both species.
- Each sentence must stand alone, explicitly contain its species name, and be
  at most 28 English words.
- Use only visible morphology: colour, pattern, body-part location, shape, or
  proportion.
- Do not use habitat, geography, season, behaviour, sound, taxonomy, rarity,
  or the image's presumed label.
- Avoid vague words such as "distinctive", "typical", "usually", or "may".
- Do not merely paraphrase the four failed prompts.
- Preserve class order [46, 47].
- Return strict JSON only, with no Markdown or explanation, in exactly this
  schema:

{
  "46_47": {
    "classes": [46, 47],
    "prompt_pairs": [
      {
        "attr_type": "short visible attribute name",
        "prompt_pair": [
          "a photograph of an american goldfinch, a type of bird, with ...",
          "a photograph of a european goldfinch, a type of bird, with ..."
        ]
      }
    ]
  }
}

The prompt_pairs array must contain exactly four objects and every sentence
must end with a period.
```
