# The submitted proposal

`AutoScope-Proposal-OpenCV-AI-Competition-2026.pdf` is what went in, on
19 August 2026. Nine pages. Both award paths: Best Use of COOL and Agentic
Vision.

`proposal.md` is the same text in markdown, split by form field.
`architecture.svg` is Figure 1 as a standalone file.
`proposal-print.html` is the source the PDF is rendered from:

```
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="AutoScope-Proposal-OpenCV-AI-Competition-2026.pdf" \
  proposal-print.html
```

## What the data has already changed

Building against the real NLM images contradicted three things in the submitted
text. `docs/data-notes.md` has the detail. In short:

1. Frames are round eyepiece views and 54% of each one is black, so the pipeline
   crops to the disc first. Not mentioned in the proposal, and it removes about
   58% of the pixels we process.
2. There are no z-stacks in the public data, so `refocus()` is simulated and
   every synthetic frame is marked as such. The proposal implies it is real.
3. A WHO 200-cell count is about 9 of these images, not 100 microscope fields.
   The saving has to be reported in white cells, not fields.

None of this weakens the project. All of it has to appear in the October report,
stated plainly, rather than quietly diverging from what we submitted.
