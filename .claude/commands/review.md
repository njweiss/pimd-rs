---
description: Phase C. Read-only, spec-keyed review of a milestone's diff
argument-hint: <milestone number>
---
You are the REVIEWER for milestone $ARGUMENTS. You did not write this code. Do not edit any file.
Print the report in the chat.

Inputs: CLAUDE.md, docs/SPEC.md, docs/MILESTONE_$ARGUMENTS.md, and `git diff main...HEAD`.

Produce, in this order:
1. **Traceability table.** For every rule ID in the brief's "In scope" table and in the spec
   sections it touches: the code that implements it (file:line) and the test that covers it (tests
   cite rule IDs in their doc comments). Mark each row COVERED, UNTESTED (code but no test) or
   MISSING (no code).
2. **Behavior not in the spec.** Anything the diff does that neither the spec nor the brief
   asks for (scope creep, invented rules, silent defaults, extra dependencies).
3. **Determinism hazards (SPEC D-2, D-3, T-1).** Check for each of these:
   - float reductions whose order can depend on thread scheduling (rayon `sum`, `reduce`,
     `par_iter().map().sum()`); force or energy scatter across threads instead of per-atom
     accumulation reduced in index order;
   - `HashMap`/`HashSet` iteration that can reach numbers or output ordering;
   - wall-clock time, addresses, thread ids, environment-dependent behavior;
   - RNG draw order differing from the spec (T-1: mode outermost, then atom, then dimension),
     randomness not taken from `pimd_core::Rng`, draws in data-dependent or unspecified order,
     per-thread RNG streams;
   - float operations in a different order than the spec writes them (B A O A B, A applied for
     dt/2 twice, forces reused rather than recomputed);
   - f32 anywhere outside model adapters.
4. **Physics-convention audit.** Verify against the spec, line by line:
   ω_P = P/β (RP-2); mode frequencies and column order (RP-4); momentum variance m·P/β (RP-3, T-1);
   γ_k = 2ω_k, c_k = exp(−γ dt) (T-1, T-2); the 1/P in E-1; forces exclude springs (S-3); ring
   positions never wrapped by integrators (C-5); unit constants and conversions only in `units`
   (U-3); `subset` handling (P-4).
5. **Test integrity.** `git diff <test commit> -- ':(glob)crates/*/tests/**'` must be empty and there
   must be no new files there; flag any `#[ignore]`, loosened tolerance, special-casing of test
   inputs in non-test code, or reference-data regeneration that changed committed values.
6. **Robustness.** `unwrap`/`expect`/indexing/arithmetic that can panic or overflow on reachable
   inputs in non-test code; divisions by ω_k = 0, by zero mass, or by zero cell length.
7. **Verdict.** A list of findings, each with severity (blocking / should-fix / nit), file:line,
   and the spec sentence it relates to. Be specific; say "no findings" for a section that has
   none. Do not pad. Do not propose rewrites of working code.

Treat the spec as the authority. If code and spec disagree, that is a finding even if the tests
pass.
