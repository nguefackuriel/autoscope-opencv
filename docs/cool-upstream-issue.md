# The COOL upstream bug report

The body is generated, not written:

    python bench/upstream_issue.py

Every number in it is computed from `bench/results/` when the script runs, so
it cannot drift from the files it tells the reader to check. The last generated
copy is `docs/cool-upstream-issue-body.md`.

## Why it is generated

The first version of this was typed by hand from a benchmark run, and then the
benchmark was re-run. An independent check before posting found that every
millisecond in the main table had quietly become a number no file in
`bench/results/` contained any more, while the report's own opening sentence
promised the opposite. It also found a swapped baseline (`medianBlur` 23x is
against a build with the Arm HAL compiled out, not against the wheel: against
the wheel it is 1.22x), a stage count that was wrong in the direction that
flattered the product being reported, and the same measurement quoted twice
with two values.

None of that was posted. The script exists so that class of error cannot
happen again.

## Where it went

**Posted 17 October 2026: https://github.com/opencv/opencv/issues/30373**

The OpenCV tracker: the dispatch question is theirs, and they can
route it. KleidiCV's own tracker is Arm's GitLab,
<https://gitlab.arm.com/kleidi/kleidicv>, which needs an Arm account. The AWS
Marketplace listing for "Cloud Optimized OpenCV for AWS Graviton4" has a
support contact for the image itself.

## What it claims, and what it does not

It claims: on the Marketplace image, large-kernel filter and morphology calls
at 12 MP take the same time on 32 threads as on 1, while the PyPI wheel gains
15 to 26 times on the same calls on the same instance type; at one thread the
two builds are within 3% of each other, so the kernels are the same speed;
banding the call by hand recovers 16x; and both builds return identical counts.

It does not claim to know the cause. Four things differ between the two builds
at once, the image cannot be rebuilt from outside, and the report says so and
asks a question instead.
