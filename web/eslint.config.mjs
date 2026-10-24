import next from "eslint-config-next/core-web-vitals";

// `api-types.ts` is generated from the committed contract; linting it would report on a file
// nobody edits by hand.
const config = [
  ...next,
  {
    ignores: [
      ".next/**",
      "node_modules/**",
      "playwright-report/**",
      "test-results/**",
      "next-env.d.ts",
      "src/lib/api-types.ts",
    ],
  },
];

export default config;
