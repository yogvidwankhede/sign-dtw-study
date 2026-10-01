# Contributing

Issues and pull requests are welcome. Please:

- Run `python -m pytest -q tests`. If a change alters any number, rerun `python -m signstudy.report` and commit the regenerated
  files; CI fails if the committed tables differ from what the committed per-query results produce.
- Do not edit `PREREG.md` after the fact. Changes of plan are listed under its "Deviations" heading with a date and a reason.
- Do not add videos, face landmarks or any data you are not allowed to share. PopSign terms are in `DATA.md`.
- Test-split commands (`--split test`) refuse to run unless the git tag `prereg-v1` exists and the code is committed; do not
  bypass this. Use `--split val` for development.
- The project studies isolated-sign recognition only. Please do not describe it as translation or interpretation, or as
  helping Deaf people, unless you have evidence from Deaf users.
- If an AI assistant wrote part of your change, say so in the pull request.

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
