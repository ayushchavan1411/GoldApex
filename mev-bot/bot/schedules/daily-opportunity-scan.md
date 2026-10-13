---
cron: 0 13 * * *
---
Run a read-only opportunity scan across the owner's chosen chains/markets (default: least-competed Morpho Blue-style lending liquidations). Rank by estimated net profit after gas using the owner's minimum profit rule. Do not submit transactions. Summarize top candidates and recommend whether a fork test is warranted.
