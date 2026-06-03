# jabla — Makefile wrapping the jank CLI.
#
# There is no Leiningen/deps "build" for jank yet; the workflow is the `jank`
# CLI + the Clojure CLI (only to compute --module-path). This Makefile is the
# substitute build tool. See `make help`.

JANK        ?= jank
CLOJURE     ?= clojure
MAIN_NS     ?= jabla.core
TEST_RUNNER ?= jabla.test-runner

# jank requires --module-path. With no external deps, src:test is correct and
# needs no JDK/Clojure-CLI. Once you add deps to deps.edn, run `make module-path`
# and pass the result: `make repl MODULE_PATH="$(make -s module-path)"`.
MODULE_PATH ?= src:test

.PHONY: help repl run test compile clean module-path check-jank check-clojure doctor health

help:
	@echo "jabla targets:"
	@echo "  make repl         Start a jank REPL on the module path (then connect Conjure/CIDER via nREPL)"
	@echo "  make run          Run $(MAIN_NS) -main"
	@echo "  make test         Run the hand-rolled test runner ($(TEST_RUNNER))"
	@echo "  make compile      AOT-compile $(MAIN_NS)  [verify exact subcommand against your jank version]"
	@echo "  make module-path  Print the Clojure-CLI-computed module path (needs clojure + JDK)"
	@echo "  make doctor       Report which tools are installed"
	@echo "  make health       Run 'jank check-health' (jank's own install diagnostic)"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "  Override vars: JANK=, CLOJURE=, MAIN_NS=, MODULE_PATH="

check-jank:
	@command -v $(JANK) >/dev/null 2>&1 || { \
	  echo ">> '$(JANK)' not found. jank isn't on PATH — see docs/jank-notes.md (Install)."; exit 1; }

check-clojure:
	@command -v $(CLOJURE) >/dev/null 2>&1 || { \
	  echo ">> '$(CLOJURE)' not found (needs a JDK). Only required to recompute module-path."; exit 1; }

repl: check-jank
	$(JANK) --module-path $(MODULE_PATH) repl

run: check-jank
	$(JANK) --module-path $(MODULE_PATH) run-main $(MAIN_NS)

test: check-jank
	$(JANK) --module-path $(MODULE_PATH) run-main $(TEST_RUNNER)

# AOT — jank can emit statically/dynamically linked executables. `compile-module`
# AOT-compiles a namespace + its deps; `jank compile` builds a project whose
# entrypoint module has -main (see `jank --help`).
compile: check-jank
	$(JANK) --module-path $(MODULE_PATH) compile-module $(MAIN_NS)

module-path: check-clojure
	@$(CLOJURE) -A:test -Spath

doctor:
	@printf "%-10s " "jank:";    command -v $(JANK)    >/dev/null 2>&1 && echo "found ($$(command -v $(JANK)))" || echo "NOT FOUND"
	@printf "%-10s " "clojure:"; command -v $(CLOJURE) >/dev/null 2>&1 && echo "found"                          || echo "NOT FOUND (only needed for module-path)"
	@printf "%-10s " "make:";    make --version | head -1

health: check-jank
	$(JANK) check-health

clean:
	rm -rf target classes ./jabla *.o *.so *.dylib *.ll *.bc
