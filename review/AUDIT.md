# README Audit — plato-training

**Date:** 2026-05-17 | **Reviewer:** Forgemaster ⚒️

## Scores

| Criterion | Score | Notes |
|-----------|:-----:|-------|
| WHAT it is | ✅ | "Train, compress, and deploy micro models for PLATO rooms" |
| WHY you'd use it | ⚠️ | Fleet results table is impressive but assumes reader knows what "PLATO rooms" and "micro models" are |
| HOW to install | ❌ | No install command. Dependencies table mentions other packages but no pip/cargo command. |
| HOW to use (code) | ✅ | Good Quick Start + Ensign Interface examples |
| Links / context | ⚠️ | Links to dependency repos in table, but no link to SuperInstance org or ecosystem docs |

**Total: 3.5/5**

## Issues

1. **No install command.** Needs `pip install plato-training` (or instructions for dev install).
2. **Weak "Why".** The fleet results table is a great selling point, but it needs a 1-sentence intro like "Ship 100% accuracy drift detection to any hardware in one function call."
3. **Missing ecosystem links.** Should link to plato-model-ocean, plato-escalation-gate for the full intelligence stack.

## Action Taken

- ✅ README rewritten with install, why, and links sections added
