import { defineAgent } from "@cursor/bdk";

export default defineAgent({
  name: "MEV Bot",
  description: "Finds under-competed on-chain profit opportunities like flash-loan liquidations, backtests them, and runs small, capped bots with daily reports. No sandwiching.",
  model: {
    id: "grok-4.5",
    params: [
      { id: "effort", value: "high" },
      { id: "fast", value: "true" },
    ],
  },
  hosting: {
    secretNames: ["ALCHEMY_API_KEY", "BURNER_PRIVATE_KEY", "RPC_URL"],
  },
});
