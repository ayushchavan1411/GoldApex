---
name: finish-grokbot-export
description: "Finish moving mev-bot off Grok Bot: secrets, connector sign-ins, the Slack app, and deploy. Use when asked to finish, deploy, or cut over this exported agent."
---
# Finish the mev-bot export

@cursor/bdk@0.2.24 generated this project from a Grok Bot recipe. `grokbot-export.json` records the recipe and every answer.

1. Set the secrets in `SECRETS.md`.
2. Run `bdk validate`, then `bdk dev`, and try the agent in the playground.
3. Push the project to a `ayushchavan1411/GoldApex` repository; the export does not create it.
4. Deploy with `bdk deploy` once the steps above check out.

## Not carried over

- `profile.avatar`: BDK agents have no avatar; kept in grokbot-export.json for the Slack app icon.
