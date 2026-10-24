# House style for anything people outside the team will read

Set by Uriel on 24 October 2026, after two rewrites of the video script still
read as machine-written. `docs/check_script_style.py` enforces the parts of
this that a regular expression can see. The rest is judgement.

The rules apply to the report, the demo page, the video script and anything
posted in the team's name. They do not apply to code comments or to the
`docs/results-*.md` laboratory notes, which are written for ourselves.

1. Mix short, direct sentences with long ones carrying several clauses. A page
   where every sentence runs to the same length reads as generated, whatever
   the words are.
2. Use present participle clauses sparingly, at most once per 500 words.
3. No contractions, no slang, no spoken register.
4. Mix active and passive voice. Depending on one alone flattens the rhythm.
5. Replace inflated verbs. Not *enhance*, *boost*, *elevate*, *ensure* or
   *crucial*; plain everyday words instead.
6. No phrasal verbs and no idiom. Say the thing directly.
7. Vary how sentences start, and avoid repeating a word close to itself.
8. Avoid listing three items. Drop one if the meaning survives, or move it into
   a second sentence.
9. State what something is. Do not first state what it is not, and do not use
   *rather than* or *it is not X, it is Y*.
10. Never write that something *extends beyond* anything.
11. Mix simple, compound and complex sentences. Use coordination and
    subordination both.
12. Meta-discourse is allowed where it helps: *as mentioned earlier*, *to
    summarise*. Paragraph-level summaries and conclusions are not.
13. Vary paragraph length as well as sentence length, and break long passages
    up.
14. No inflated claims without a measurement behind them. Nothing *reshapes*
    or *fundamentally changes* anything.
15. No drama, no suspense, no hooks, no metaphors.
16. Open with evidence, not with a general statement about the field.

## Running the check

    python docs/check_script_style.py

It reports every hit with the sentence around it and exits non-zero, so the
text cannot drift back. The check is only as good as its patterns: an early
version reported zero participle clauses on a text that had four, and counted
*wheel's 547* as a contraction. Treat a pass as the floor, not the ceiling.
