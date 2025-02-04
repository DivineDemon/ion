import { defineConfig } from "eslint-define-config";

export default defineConfig({
  parser: "@typescript-eslint/parser",
  extends: [
    "next/core-web-vitals",
    "next/typescript",
    "prettier",
    "plugin:@typescript-eslint/recommended-type-checked",
    "plugin:@typescript-eslint/stylistic-type-checked",
  ],
  plugins: ["check-file", "n", "@typescript-eslint"],
  rules: {
    "@typescript-eslint/array-type": "off",
    "@typescript-eslint/consistent-type-definitions": "off",
    "@typescript-eslint/consistent-type-imports": [
      "warn",
      {
        prefer: "type-imports",
        fixStyle: "inline-type-imports",
      },
    ],
    "@typescript-eslint/no-unused-vars": [
      "warn",
      {
        argsIgnorePattern: "^_",
      },
    ],
    "@typescript-eslint/require-await": "off",
    "@typescript-eslint/no-misused-promises": [
      "error",
      {
        checksVoidReturn: {
          attributes: false,
        },
      },
    ],
    "no-console": ["error"],
    eqeqeq: ["error"],
    "prefer-arrow-callback": ["error"],
    "prefer-template": ["error"],
    semi: ["error"],
    quotes: ["error", "double"],
    "n/no-process-env": ["error"],
    "check-file/filename-naming-convention": [
      "error",
      {
        "**/*.{ts,tsx}": "KEBAB_CASE",
      },
      {
        ignoreMiddleExtensions: true,
      },
    ],
    "check-file/folder-naming-convention": [
      "error",
      {
        "src/**/!^[.*": "KEBAB_CASE",
      },
    ],
  },
});