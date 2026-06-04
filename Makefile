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

# Local C++ syntax/type check for cpp/ interop headers (no jank, no link step).
# For headers that #include <cblas.h>, append the OpenBLAS include dir, e.g.:
#   make cpp-check CPP_CHECK_FLAGS="-std=c++20 -Wall -Wextra -Icpp/include -I$$(brew --prefix openblas)/include"
# CXX is a make built-in (often 'c++' = Apple clang, which can't find libc++ on a
# CLT-only mac); override the built-in default but respect an explicit CXX=... .
ifeq ($(origin CXX),default)
CXX := clang++
endif
CPP_CHECK_FLAGS ?= -std=c++20 -Wall -Wextra -Icpp/include

.PHONY: help repl run test compile clean module-path check-jank check-clojure doctor health lint-ascii lint hooks deps cpp-check check

help:
	@echo "jabla targets:"
	@echo "  make repl         Start a jank REPL on the module path (then connect Conjure/CIDER via nREPL)"
	@echo "  make run          Run $(MAIN_NS) -main"
	@echo "  make test         Run all test suites (clojure.test). One suite: make test SUITE=autograd|blas"
	@echo "  make compile      AOT-compile $(MAIN_NS)  [verify exact subcommand against your jank version]"
	@echo "  make module-path  Print the Clojure-CLI-computed module path (needs clojure + JDK)"
	@echo "  make doctor       Report which tools are installed"
	@echo "  make health       Run 'jank check-health' (jank's own install diagnostic)"
	@echo "  make lint-ascii   Fail if any .jank source has non-ASCII bytes (lexer limitation)"
	@echo "  make lint         Run clj-kondo over the .jank sources (project-wide)"
	@echo "  make cpp-check    Syntax/type-check cpp/ headers locally with clang (pre-devbox)"
	@echo "  make check        Run all static checks: lint-ascii + lint + cpp-check"
	@echo "  make hooks        Install the git pre-commit hook (.githooks)"
	@echo "  make deps         Install native C++ deps (BLAS, ...) via apt (Linux) / brew (macOS)"
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

run: lint-ascii check-jank
	$(JANK) --module-path $(MODULE_PATH) run-main $(MAIN_NS)

test: lint-ascii check-jank
	$(JANK) --module-path $(MODULE_PATH) run-main $(TEST_RUNNER) $(if $(SUITE),-- $(SUITE))

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
# they reach the devbox JIT. Header-only (no link), so no libs needed unless a
# header pulls in <cblas.h> (then extend CPP_CHECK_FLAGS; see above).
cpp-check:
	@cxx="$(CXX)"; cflags="$(CPP_CHECK_FLAGS)"; \
	if [ "$$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then \
	  llvm="$$(brew --prefix llvm 2>/dev/null)"; \
	  [ "$$cxx" = "clang++" ] && [ -x "$$llvm/bin/clang++" ] && cxx="$$llvm/bin/clang++"; \
	  ob="$$(brew --prefix openblas 2>/dev/null)"; \
	  [ -d "$$ob/include" ] && cflags="$$cflags -I$$ob/include"; \
	fi; \
	command -v "$$cxx" >/dev/null 2>&1 || { echo ">> '$$cxx' not found (install clang/LLVM; on macOS: brew install llvm)."; exit 1; }; \
	echo ">> using $$cxx"; \
	found=0; status=0; \
	for h in cpp/include/*.hpp; do \
	  [ -e "$$h" ] || continue; found=1; echo ">> checking $$h"; \
	  echo "#include <$$(basename $$h)>" | "$$cxx" $$cflags -x c++ -fsyntax-only - || status=1; \
	done; \
	[ "$$found" = 1 ] || echo "cpp-check: no headers in cpp/include/"; \
	exit $$status

# Full static-check sweep, used by the pre-commit hook (and suitable for CI). Just
# composes the independent targets -- lint-ascii / lint / cpp-check stay usable on
# their own. Needs the C++ toolchain (clang + OpenBLAS headers) for cpp-check.
check: lint-ascii lint cpp-check
	@echo "check: all passed"

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
