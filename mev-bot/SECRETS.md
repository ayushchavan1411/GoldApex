# Secrets

The project reads these secrets by name. The Grok Bot recipe carried no values, so set each one before you run the agent.

| Name | Read by |
| --- | --- |
| `ALCHEMY_API_KEY` | your own code (named in the answers) |
| `BURNER_PRIVATE_KEY` | your own code (named in the answers) |
| `RPC_URL` | your own code (named in the answers) |

Local runs read secrets from the environment. A deployment reads the values you store with `bdk secrets set mev-bot NAME...`.
