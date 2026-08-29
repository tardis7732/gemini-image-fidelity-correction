# Ungated single-run appendix

This folder preserves the first paired recurrence run before the low-deformation applicability gate was introduced.

At corrected-chain pass 2, Gemini changed the immediate input substantially: input-to-output MAE similarity was `97.9543%`, while the normal no-change passes were approximately `99.17%` to `99.25%`. The correction model was still applied in that original run, producing a final corrected-chain similarity of `97.1766%` versus `96.4261%` for the raw chain.

These files are retained for auditability. The main report uses the predeclared `99.0%` immediate-similarity gate because the correction model targets low-deformation no-change outputs rather than large semantic regeneration.
