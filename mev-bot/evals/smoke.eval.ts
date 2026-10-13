import { defineEval } from "@cursor/bdk/evals";

export default defineEval({
  description: "The agent answers a first question.",
  async test(t) {
    await t.send("What can you help me with?");
    t.succeeded();
  },
});
