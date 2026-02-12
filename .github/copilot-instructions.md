<!-- Copilot / AI agent guidance for this repository -->

# Copilot instructions — Fabiola

Purpose: give concise, project-specific guidance so an AI coding agent can be immediately productive.

- **Repo snapshot:** single Python script at `main.py` whose current contents are a simple entrypoint printing "hello". No `requirements.txt`, `pyproject.toml`, `README.md`, tests, or CI config detected.

- **Primary entrypoint:** `main.py` — run locally with:

```bash
python main.py
```

- **Big picture:** extremely small single-file project. When expanding the project, prefer adding clear modules under the repo root (e.g., `app/`, `lib/`) and keep `main.py` as the thin entrypoint that imports from those modules.

- **Patterns & conventions observed (use these as source of truth):**
  - Single-script layout: keep top-level orchestration in `main.py` and move reusable logic into new modules.
  - No dependency manifest: if adding external libraries, create `requirements.txt` (for simple scripts) or `pyproject.toml` for a package.

- **What AI agents should do first when making changes:**
  1. Read `main.py` to understand the current entrypoint and global side effects.
 2. If adding behavior, create a new module under the repo root (for example, `app/processor.py`) and import it from `main.py` rather than adding large blocks to `main.py`.
 3. Add or update `requirements.txt` when new packages are introduced. Example line: `requests==2.31.0`.

- **Testing & CI:** no tests or CI found. Do not assume testing frameworks. If you add tests, include a `tests/` folder and a `requirements-dev.txt` or `pyproject.toml` entry for the test runner.

- **Files to check for context before editing:**
  - `main.py` — current entrypoint and only source file
  - repository root — look for newly added config (`requirements.txt`, `pyproject.toml`, `README.md`) to understand intended packaging

- **Integration & external dependencies:** none discovered. Treat external integrations as new additions; add explicit config files and document run steps in `README.md`.

- **Commit and PR guidance for AI agents:**
  - Keep changes small and focused per PR (one responsibility per PR).
  - When adding packages, include `requirements.txt` and an example run command in `README.md`.

- **Examples from this codebase:**
  - Current `main.py`:

```python
print("hello")
```

  - If implementing a feature, create `app/` and move logic there, then modify `main.py` to import and call the new function.

If anything here is unclear or you expect other files/configs in the workspace, tell me which sections to expand or correct and I will iterate.
