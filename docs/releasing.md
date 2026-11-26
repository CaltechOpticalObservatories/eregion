# Releasing

An eregion release is an annotated git tag named `eregion-MAJOR.MINOR.PATCH`, for
example `eregion-0.2.0`. Pushing the tag does two things:

- The package version becomes `0.2.0`. eregion has no version number written in
  the source: [setuptools-scm](https://setuptools-scm.readthedocs.io/) reads it
  from the latest `eregion-*` tag. A commit after a tag installs as a dev version
  such as `0.2.1.dev3+g1a2b3c4`.
- The **Release Announcement** GitHub Actions workflow posts the tag message to
  Slack as the release notes.

## Cutting a release

```bash
git switch main && git pull
git log --oneline eregion-0.1.0..HEAD   # what changed since the last release
                                        # (skip this for the first release)
git tag -a eregion-0.2.0                # your editor opens; write the release notes
git push origin eregion-0.2.0
```

The tag message is posted to Slack word for word, under the heading
*Eregion 0.2.0 released*. Slack formatting works in it: `*bold*`, `_italic_`,
`` `code` `` and `•` bullets.

Example tag message:

```text
• Superflat task with simple and Welford combine methods (#35)
• Vignette and cosmic ray masking tasks (#33)
• Raise a clear error when an output has no overscan pixels (#38)
```

## Rules

- Always use an annotated tag (`git tag -a`). A lightweight tag has no
  message, so the workflow fails and nothing is posted.
- Use exactly three numbers after `eregion-`. Before posting, the workflow checks
  that setuptools-scm reads the tag as the same version, and it fails if not.
- Never move or reuse a tag once it is pushed. If a release is wrong, tag a new
  patch version.

## Re-posting an announcement

If the Slack post failed, for example because the webhook secret was missing, go
to **Actions → Release Announcement → Run workflow** and enter the existing tag
(e.g. `eregion-0.2.0`).

## One-time setup

The workflow posts through a Slack
[incoming webhook](https://api.slack.com/messaging/webhooks). The webhook URL
must be stored as the repository secret `SLACK_WEBHOOK_URL` (Settings → Secrets
and variables → Actions).
