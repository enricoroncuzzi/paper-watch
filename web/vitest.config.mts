import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

// Unit tests cover the pure logic (guards, URL checks, the SSE parser, the ask reducer). They
// never render a component, so the node environment is enough and no DOM shim is needed.
export default defineConfig({
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
      // `server-only` throws outside a React Server Components bundle; the modules that import it
      // are still plain TypeScript and are exactly what the tests want to exercise.
      "server-only": fileURLToPath(new URL("./src/test/server-only.ts", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    // One worker: a few tests stub global fetch, and a shared process keeps that simple.
    fileParallelism: false,
  },
});
