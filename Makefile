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

# Native (C++) dependencies for the cpp/ interop layer (BLAS now; add CUDA/cuBLAS
# here as the layer grows). Build/run is on the Linux devbox; macOS is kept
# working for completeness even though it doesn't run jank.
APT_DEPS  ?= libopenblas-dev
BREW_DEPS ?= openblas

# C++ interop flags for the jank CLI on run/test/repl/compile: the cpp/include
# header dir + the OpenBLAS link. jank does NOT search the default linker paths,
# so -L is required even for a system lib (a bare -lopenblas fails). Override
# CBLAS_LIBDIR on non-Ubuntu/x86_64 hosts -- find it via:
#   ldconfig -p | grep libopenblas.so
# Exported so the local C++ checks (bin/cpp-check, bin/cpp-test) pick it up too;
# their compiler/flags defaults live in those scripts (run bin/cpp-toolchain).
CBLAS_LIBDIR   ?= /lib/x86_64-linux-gnu
JANK_CPP_FLAGS ?= -I cpp/include -L $(CBLAS_LIBDIR) -lopenblas
export CBLAS_LIBDIR

.PHONY: help repl run test compile clean module-path check-jank check-blas check-clojure doctor health lint-ascii lint hooks deps cpp-check cpp-test check

help:
	@echo "jabla targets:"
	@echo "  make repl         Start a jank REPL on the module path (then connect Conjure/CIDER via nREPL)"
	@echo "  make run          Run $(MAIN_NS) -main"
	@echo "  make test         Run all test suites (clojure.test). One suite: make test SUITE=autograd|tensor"
	@echo "  make compile      AOT-compile $(MAIN_NS)  [verify exact subcommand against your jank version]"
	@echo "  make module-path  Print the Clojure-CLI-computed module path (needs clojure + JDK)"
	@echo "  make doctor       Report which tools are installed"
	@echo "  make health       Run 'jank check-health' (jank's own install diagnostic)"
	@echo "  make lint-ascii   Fail if any .jank source has non-ASCII bytes (lexer limitation)"
	@echo "  make lint         Run clj-kondo over the .jank sources (project-wide)"
	@echo "  make cpp-check    Syntax/type-check cpp/ headers locally with clang (pre-devbox)"
	@echo "  make cpp-test     Compile + run cpp/test/*.cpp locally (catches C++ logic bugs)"
	@echo "  make check        Run all static checks: lint-ascii + lint + cpp-check"
	@echo "  make hooks        Install the git pre-commit hook (.githooks)"
	@echo "  make deps         Install native C++ deps (BLAS, ...) via apt (Linux) / brew (macOS)"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "  Override vars: JANK=, CLOJURE=, MAIN_NS=, MODULE_PATH=, CBLAS_LIBDIR="

check-jank:
	@command -v $(JANK) >/dev/null 2>&1 || { \
	  echo ">> '$(JANK)' not found. jank isn't on PATH — see docs/jank-notes.md (Install)."; exit 1; }

check-clojure:
	@command -v $(CLOJURE) >/dev/null 2>&1 || { \
	  echo ">> '$(CLOJURE)' not found (needs a JDK). Only required to recompute module-path."; exit 1; }

# Preflight: fail with a clear message if OpenBLAS isn't installed, instead of
# jank's opaque "Failed to load dynamic library" when jabla.tensor loads. Gated on
# run/test/compile (which load BLAS-dependent code); repl is left ungated so you
# can poke non-blas code without OpenBLAS present.
check-blas:
	@if [ -e "$(CBLAS_LIBDIR)/libopenblas.so" ] \
	   || { command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q libopenblas; } \
	   || { [ "$$(uname -s)" = Darwin ] && [ -e "$$(brew --prefix openblas 2>/dev/null)/lib/libopenblas.dylib" ]; }; then \
	  :; \
	else \
	  echo ">> OpenBLAS not found (needed by jabla.tensor). Run 'make deps' to install it."; exit 1; \
	fi

repl: check-jank
	$(JANK) $(JANK_CPP_FLAGS) --module-path $(MODULE_PATH) repl

run: lint-ascii check-jank check-blas
	$(JANK) $(JANK_CPP_FLAGS) --module-path $(MODULE_PATH) run-main $(MAIN_NS)

test: lint-ascii check-jank check-blas
	$(JANK) $(JANK_CPP_FLAGS) --module-path $(MODULE_PATH) run-main $(TEST_RUNNER) $(if $(SUITE),-- $(SUITE))

# jank's lexer (0.1-alpha) rejects non-ASCII bytes even inside comments/strings.
# Portable guard (works with GNU and BSD grep — no -P): flag any byte outside
# printable ASCII in .jank sources before they reach the compiler.
lint-ascii:
	@if LC_ALL=C grep -rn '[^ -~]' src test --include='*.jank' >/dev/null 2>&1; then \
	  echo ">> non-ASCII found in .jank sources (jank's lexer rejects it):"; \
	  LC_ALL=C grep -rn '[^ -~]' src test --include='*.jank'; \
	  exit 1; \
	fi

# Static analysis via clj-kondo (treats .jank as Clojure). See bin/lint.
lint:
	@bin/lint

# Point git at the tracked hooks dir so the pre-commit hook runs for everyone.
hooks:
	@git config core.hooksPath .githooks && echo "git hooks installed (core.hooksPath=.githooks)"

# Native C++ dependencies for the cpp/ interop layer (see cpp/README.md). Run on
# the devbox before the BLAS spike. Extend APT_DEPS / BREW_DEPS as the layer grows.
deps:
	@case "$$(uname -s)" in \
	  Linux)  echo ">> installing via apt: $(APT_DEPS)";  sudo apt-get update && sudo apt-get install -y $(APT_DEPS) ;; \
	  Darwin) echo ">> installing via brew: $(BREW_DEPS)"; brew install $(BREW_DEPS) ;; \
	  *)      echo ">> unknown OS; install manually: $(APT_DEPS)"; exit 1 ;; \
	esac

# Local pre-flight: syntax/type-check the cpp/ interop headers with clang before
# they reach the devbox JIT (logic in bin/cpp-check).
cpp-check:
	@bin/cpp-check

# Compile + run the cpp/ unit tests locally with doctest (logic in bin/cpp-test).
cpp-test:
	@bin/cpp-test

# Full static-check sweep, used by the pre-commit hook (and suitable for CI). Just
# composes the independent targets -- lint-ascii / lint / cpp-check stay usable on
# their own. Needs the C++ toolchain (clang + OpenBLAS headers) for cpp-check.
check: lint-ascii lint cpp-check
	@echo "check: all passed"

# AOT — jank can emit statically/dynamically linked executables. `compile-module`
# AOT-compiles a namespace + its deps; `jank compile` builds a project whose
# entrypoint module has -main (see `jank --help`).
compile: check-jank check-blas
	$(JANK) $(JANK_CPP_FLAGS) --module-path $(MODULE_PATH) compile-module $(MAIN_NS)

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
