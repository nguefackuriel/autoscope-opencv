# How we work on this repo

Two people, nine weeks, and a `main` that has to stay demo-ready for the judges
at any moment. The rules below exist so experiments never collide and nothing
half-finished lands where a judge might see it.

## Branches

```
main        what we would show a judge today. Nobody pushes here directly.
develop     integration. Both of us merge here first. The default branch.
u/<topic>   Uriel's working branches      e.g. u/parasite-detector
b/<topic>   Bruno's working branches      e.g. b/graviton-container
```

One branch per piece of work, named for the work, not for the week. Delete it
after the merge. The `u/`/`b/` prefixes mean you always know whose branch you
are looking at, and neither of us ever works on the other's branch without
saying so.

GitHub's branch protection needs a paid plan on private repos, so `main` is
protected by convention instead: **nobody pushes to `main`, ever.** It only
moves by merging `develop` into it when `develop` is green and demo-ready.

## The loop

```
git checkout develop && git pull
git checkout -b u/my-topic          # or b/my-topic
# ...work, commit...
git push -u origin u/my-topic
gh pr create --base develop --fill
```

Merge your own PR when tests pass; ask for a review when the change touches
the other person's area or a number we publish. Keep PRs small enough that the
other person can actually read them.

Before opening the PR, rebase on develop so conflicts happen on your branch,
not in the merge:

```
git fetch origin && git rebase origin/develop
```

## Experiments

Experiments are branches like anything else, with one extra rule: **a failed
experiment still gets written up.** Put what you tried and what the numbers
said in `docs/`, merge the write-up, delete the code branch. We have already
been saved twice by notes on ideas that did not work; the note costs a
paragraph, re-discovering the dead end costs a week.

An experiment only merges into `develop` when its numbers beat the current
ones **on the held-out patients** (`eval/wbc_eval.py --holdout`), not on the
fit set.

## Two hard rules that protect the results

1. **The Tanzania set stays untouched.** No threshold fitted on it, no model
   trained on it, no peeking at its numbers until the pipeline is frozen. The
   generalisation claim in the final report depends on this.
2. **Published numbers come from held-out patients.** The fit set flatters by
   construction; we quote it only next to the held-out figure, never alone.

## Who owns what

From the proposal, so review requests have an obvious address:

- **Uriel**: OpenCV pipeline (`src/autoscope/`), evaluation (`eval/`), data.
- **Bruno**: AWS and infra (`infra/`), the Graviton/COOL benchmark (`bench/`),
  the Bedrock agent wiring.
- Report, demo page and video: both.
