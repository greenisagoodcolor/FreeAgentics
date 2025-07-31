## Claude Code Development Playbook (v2 – ≈ 34 500 chars, July 2025)

> **Purpose**  A single, living reference that merges every Claude‑prompt rule set plus new best‑practice additions (observability, zero‑trust, memory tuning, demo mode, incident response, etc.).
> **Audience**  AI agents *and* human devs onboarding to the project.
> **Tone**  Practical, example‑heavy, opinionated but adaptable.

---

### 0  Executive Snapshot

| Core Value         | Rule of Thumb                           | Memorable Metric               |
| ------------------ | --------------------------------------- | ------------------------------ |
| **Correctness**    | Strict **TDD** → red → green → refactor | 100 % tests green before merge |
| **Quality Gates**  | Lint + TypeCheck + SecScan + Bench      | 0 ✗ tolerated                  |
| **Simplicity**     | KISS, YAGNI, no premature optimisation  | PR ≤ 400 LOC                   |
| **Security‑first** | Zero‑trust network + mTLS + SAST/DAST   | JWT exp 15 min                 |
| **Observability**  | Structured logs + traces + metrics      | P95 API < 200 ms               |
| **Performance**    | Measure before tuning, profile in CI    | < 50 ms agent spawn            |
| **Documentation**  | Update `CLAUDE.md` *every* commit       | 1‑sheet onboarding             |

---

### 1  Daily Flow (5‑step loop)

1. **Pick task**   `task-master next` → `show` → mark *in‑progress*.
2. **Plan**   Add TODO list + failing tests; ask Qs if ambiguous.
3. **Code**   Implement smallest slice to pass test; stay immutable.
4. **Refactor & Validate**   Run `make fmt && make test && make lint && make sec && make bench`.
5. **Document & Commit**   Conventional commit, update Task Master, record insight.

> **Reality Checkpoints**  After each green run, before new feature, when hooks fail, and at EOD.

---

### 2  Task‑Master Cheat‑Sheet (top 12 commands)

```bash
# Navigation
next_task           # == task-master next
get_task 2.3        # == task-master show 2.3
set_task_status 2.3 done

# Planning & research
add_task "migrate Redis" --research
expand_task 3 --research --force
analyze_project_complexity --research

# House‑keeping
validate_dependencies
complexity_report
generate            # regenerate markdown files
```

*Always specify `--research` to let Claude pull external docs where gaps exist.*

---

### 3  Coding Standards

#### 3.1 Types & Schemas

* Source of truth = Zod schemas (runtime) → `z.infer` types (compile‑time).
* **Forbidden** `any`, broad `unknown` casts, `@ts-ignore` w/o comment.
* Use domain‑specific branded types: `type UserId = string & {readonly __brand:'UserId'}`.

#### 3.2 Functions

* ≤ 50 lines, 1 responsibility, early returns.
* Options objects for >2 args or optional arg.
* Never mutate inputs; return new objects.

#### 3.3 Naming

* camelCase variables/functions, PascalCase types, kebab‑case filenames.
* Constants: `UPPER_SNAKE_CASE` if truly constant.

#### 3.4 Comments

* Avoid “what” comments; use code. When needed, explain **why**.
* Public API exported funcs/classes get TSDoc.

---

### 4  Testing Doctrine

1. **Red** Write behaviour test via public API (Jest/Vitest + RTL + MSW).
2. **Green** Minimal code to pass.
3. **Refactor** Improve names/structure; tests stay green.
4. **Coverage Goal** 100 % meaningful coverage *via behaviour*, no test‑code coupling.
5. **Factories** `getMock*` helpers validate against real schema; allow `Partial<T>` overrides.

> **Pitfall** Writing more than one failing test before coding breaks tight loop – don’t.

---

### 5  CI/CD Quality Gates

| Stage              | Command                            | Blocks Merge if…        |
| ------------------ | ---------------------------------- | ----------------------- |
| **Fmt**            | `make fmt`                         | file not auto‑formatted |
| **Lint**           | `make lint` (ruff/eslint)          | any warning             |
| **Types**          | `make typecheck`                   | TS error                |
| **Security**       | `make sec` (Bandit/Semgrep/Safety) | severity ≥ MEDIUM       |
| **Tests**          | `make test`                        | failure or <100 % cov   |
| **Benchmark**      | `make bench`                       | >10 % regression        |
| **Container scan** | `make scan`                        | critical CVE            |

Green‑all or merge‑none.

---

### 6  Security Architecture (Zero‑Trust)

1. **mTLS everywhere** (RS256 certs, auto‑rotate ≤ 90 d, 10 ms issuance via cache).
2. **Istio mesh** Sidecars enforce policy; TLS 1.3 only, modern ciphers.
3. **Identity‑Aware Proxy** Validates JWT *and* session risk score on every hop.
4. **Field‑Level Encryption** AWS KMS / Vault; decorator pattern; rotate keys w/o downtime.
5. **Quantum‑ready** Use Kyber KEM + Dilithium hybrid for long‑lived secrets.
6. **SOAR playbooks** YAML‑defined auto‑triage; runbooks kept in repo; metrics in Grafana.

---

### 7  Observability & Telemetry

* **Logging** JSON lines, ISO‑8601, include `trace_id`, `span_id`, `user_id`.
* **Tracing** OpenTelemetry; sample 1 % in prod, 100 % on errors.
* **Metrics** RED + USE + custom biz metrics; expose `/metrics` Prom endpoint.
* **Dashboards** Grafana folders per domain; P95 latency alert ≥ 200 ms.
* **Incident Signals** PagerDuty rules: Sev‑1 page < 2 min, Sev‑2 slack < 10 min.

---

### 8  Performance & Memory Optimisation

| Technique                                              | Win              | When to apply              |
| ------------------------------------------------------ | ---------------- | -------------------------- |
| Sparse matrices (scipy.sparse)                         | 95–99 % mem cut  | Belief arrays >90 % zeros  |
| Object pools                                           | avoid GC churn   | Hot (>10 k/s) obj creation |
| Shared mem segments                                    | Σmem ↓ 40 %      | Large read‑only config     |
| Lazy init                                              | startup ↓ 200 ms | Rarely used subsystems     |
| Profilers – `tracemalloc`, `memory_profiler`, `py-spy` | locate leaks     | Weekly CI job              |

Memory budget per agent = 34.5 MB hard cap; alert at 80 %.

---

### 9  Data‑Layer & Database Optimisation

* **Pool sizing** = `(num_cores * 2) + effective IO wait` formula.
* **Isolation** SERIALIZABLE only for money moves; READ\_COMMITTED elsewhere.
* **Indexing** Partial indexes on `deleted_at IS NULL`; compound `(workspace_id, updated_at)`.
* **Query Plan Checks** `EXPLAIN ANALYZE` in CI for queries > 30 ms.
* **SQLite tests** `PRAGMA foreign_keys=ON`; fixtures create fresh engine per test.

---

### 10  Resilience Patterns

* **Circuit Breaker** CLOSED→OPEN @ 5 failures / 30 s; HALF\_OPEN probe 1 req.
* **Retry** Exponential (100 ms→1.6 s, jitter 30 %).
* **Bulkhead** Dedicated asyncio queues per external API.
* **Graceful Degradation** Return cached or stubbed result; never crash UI.

---

### 11  Demo Mode & Mock Providers

* If `DATABASE_URL` absent ⇒ log ⚠️ *demo mode*; use in‑mem store.
* Mock LLM provider returns deterministic small embeddings + canned GMN prompts.
* Demo WebSocket at `/api/v1/ws/demo` - no auth required, simulates agent ops.
* Frontend auto-connects to demo endpoint when `NEXT_PUBLIC_WS_URL` not set.
* Ensure *full happy path* passes e2e tests w/o secrets.

---

### 12  QA Failure Cookbook (Top 5)

| Symptom                        | Root Cause                   | Fix                                          |
| ------------------------------ | ---------------------------- | -------------------------------------------- |
| **CI “ResolutionImpossible”**  | incompatible transitive deps | add range pin `<` next major                 |
| **Test flake on Sunday 00:00** | timezone/UTC bug             | freeze time in test; use `pendulum`          |
| **Table redefinition**         | duplicate SQLAlchemy Base    | import shared `Base`; `extend_existing=True` |
| **"NoneType has no attr F"**   | PyMDP API drift              | add version guards; default fallback         |
| **Ruff F841**                  | unused var                   | delete or log var; never ignore              |

---

### 13  Incident Response 5‑step

1. Detect (alert)  → PagerDuty Sev level.
2. Contain  → flip feature flag, scale down, or break‑glass commit.
3. Eradicate  → root‑cause via traces/metrics; write failing regression test.
4. Recover  → deploy hotfix; validate green gates.
5. Learn  → 1‑page post‑mortem ≈ 48 h; update playbook & runbook.

---

### 14  Multi‑Agent / Committee Workflow

* Spawn analysis agent, test‑writer agent, implementer agent in parallel.
* **Nemesis‑style Debate Cycle** per commit:

  1. **Kent Beck** – Are tests minimal & expressive?
  2. **Martin Fowler** – Is code refactored for clarity?
  3. **Jessica Kerr** – Is outcome observable?
  4. **Charity Majors** – Will prod signals catch failure?
* Resolve dissent before merge; document decision.

---

### 15  Commit & PR Guidelines (Expanded)

```
feat(auth): add JWT rotation (task 3.2)
fix(db): handle txn rollback on SIGTERM (task 2.4)
refactor(payment): extract shipping cost helper
perf(agent): spawn bar drops from 75→42 ms
```

* **One idea per commit**, tests included.
* Draft PR allowed for early feedback but must be green before “Ready”.
* Reviewers checklist: naming, tests, observability, security, docs.

---

### 16  Logging & Tracing Examples

```ts
// logger.ts
authedLogger.info(
  {
    trace_id,
    user_id: ctx.user.id,
    op: "checkout", amount: ctx.amount,
  },
  "submitted payment"
);
```

```yaml # open‑telemetry span example
- name: payment.authorize
  start_time: 2025-07-29T12:00:00Z
  attributes:
    user.id: "u_123"
    payment.id: "pay_456"
    amount: 42.00
    currency: "GBP"
```

---

### 17  CLI & GH Commands Reference

```bash
# CI logs
gh run list --limit 10
gh run view 123 --log-failed

# Fix & re‑push quick patch
git restore src/payment.ts
npm run test:watch  # keep loop green

# Benchmark diff
hyperfine 'npm run bench' --export-markdown bench.md
```

---

### 18  Glossary

| Term        | Meaning                                           |
| ----------- | ------------------------------------------------- |
| **TDD**     | Test‑Driven Development                           |
| **RED/USE** | Rate, Errors, Duration / Util, Saturation, Errors |
| **SoR**     | System of Record                                  |
| **LLM**     | Large Language Model                              |
| **SOAR**    | Security Orchestration, Automation & Response     |
| **mTLS**    | Mutual TLS                                        |

---

### 19  Appendix A – Expert Bios (short‑form)

* **Kent Beck**  Inventor of TDD & Extreme Programming; coach at Facebook; says “make it work, make it right, make it fast.”
* **Robert C Martin** (Uncle Bob)  Clean Code evangelist; SOLID principles.
* **Martin Fowler**  Chief Scientist at ThoughtWorks; author *Refactoring*; domain‑driven design advocate.
* **Michael Feathers**  Wrote *Working Effectively with Legacy Code*; specialises in seams & tests for legacy systems.
* **Jessica Kerr**  Observability & socio‑tech speaker; “software is a symmathesy.”
* **Sindre Sorhus**  Maintainer of 1 k+ OSS libs; emphasises small utilities & rigorous automated tests.
* **Addy Osmani**  Chrome performance lead; champions Core Web Vitals.
* **Sarah Drasner**  VP DevRel at Netlify; educator on Vue, animations.
* **Evan You**  Creator of Vue; pragmatic progressive enhancement.
* **Rich Harris**  Creator of Svelte; “run‑time is a bug.”
* **Charity Majors**  Honeycomb co‑founder; observability & production ownership.

Each review cycle, at least three voices critique the change before merge.

---

### 20  Appendix B – Reality‑Check Questions

1. Does a failing test *force* this code?
2. Is the chosen design the simplest that could possibly work?
3. If this fails at 02:00, will on‑call have the signals to fix fast?
4. Could we delete this feature tomorrow with minimal blast radius?
5. What future maintainer knowledge would make this clearer *now*?

---

### 21  Living Document Rule

* **Always shorter than 35 000 characters** to fit Claude context budget.
* Every substantive project lesson → add/adjust section and commit.
* Remove stale or duplicated info aggressively – clarity over nostalgia.

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
