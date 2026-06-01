#!/usr/bin/env bash
# =============================================================================
# run.sh — OP'26 End-to-End Pipeline Runner
#
# Usage:
#   chmod +x run.sh
#   export GROQ_API_KEY='your-groq-key-here'
#   ./run.sh
#
# Optional flags:
#   --steps N        Number of agentic loop steps (default: full test set ~194)
#   --delay N        Seconds between LLM calls (default: 1.0)
#   --skip-preprocess  Skip preprocessing if unified_analytical_base.csv exists
#   --skip-eda         Skip EDA plots
#   --skip-tests       Skip test suite
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
header()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${RESET}"; \
            echo -e "${BOLD}${CYAN}  $*${RESET}"; \
            echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
STEPS=""
DELAY="1.0"
SKIP_PREPROCESS=false
SKIP_EDA=false
SKIP_TESTS=false
PYTHON=".venv/bin/python"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)          STEPS="$2"; shift 2 ;;
    --delay)          DELAY="$2"; shift 2 ;;
    --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
    --skip-eda)       SKIP_EDA=true; shift ;;
    --skip-tests)     SKIP_TESTS=true; shift ;;
    *) warn "Unknown flag: $1"; shift ;;
  esac
done

# ── Sanity checks ─────────────────────────────────────────────────────────────
header "OP'26 — Pre-flight Checks"

if [[ ! -f "$PYTHON" ]]; then
  error "Virtual environment not found at .venv/. Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
fi
success "Python venv found: $PYTHON"

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  error "GROQ_API_KEY is not set.\n  Run: export GROQ_API_KEY='your-groq-key-here'\n  Get a free key at: https://console.groq.com"
fi
success "GROQ_API_KEY is set"

if [[ ! -f "data/raw/acndata_sessions.json.xlsx" ]]; then
  error "ACN data not found at data/raw/acndata_sessions.json.xlsx\n  Download from: https://ev.caltech.edu/dataset.html"
fi
success "ACN data found"

for f in volume.csv occupancy.csv duration.csv; do
  if [[ ! -f "data/raw/$f" ]]; then
    error "UrbanEV file not found: data/raw/$f\n  Download from: https://github.com/IntelligentSystemsLab/ST-EVCDP"
  fi
done
success "UrbanEV data found (volume, occupancy, duration)"

# ── Step 1: Tests ─────────────────────────────────────────────────────────────
if [[ "$SKIP_TESTS" == false ]]; then
  header "Step 1/5 — Running Test Suite"
  $PYTHON -m pytest tests/ -q --tb=short 2>&1 | tail -8
  success "All tests passed"
else
  warn "Skipping tests (--skip-tests)"
fi

# ── Step 2: Preprocessing ─────────────────────────────────────────────────────
header "Step 2/5 — Data Preprocessing"

UNIFIED="data/processed/unified_analytical_base.csv"
if [[ "$SKIP_PREPROCESS" == true && -f "$UNIFIED" ]]; then
  warn "Skipping preprocessing — $UNIFIED already exists"
else
  info "Running preprocessing pipeline..."
  $PYTHON -m src.pipeline.preprocess \
    --acn data/raw/acndata_sessions.json.xlsx \
    --urban-dir data/raw \
    --out "$UNIFIED" \
    --log-level INFO
  success "Unified analytical base written → $UNIFIED"
fi

# ── Step 3: EDA ───────────────────────────────────────────────────────────────
header "Step 3/5 — Exploratory Data Analysis"

if [[ "$SKIP_EDA" == false ]]; then
  info "Generating EDA plots..."
  $PYTHON -m src.eda.plots \
    --base "$UNIFIED" \
    --acn data/raw/acndata_sessions.json.xlsx \
    --log-level INFO
  success "EDA plots saved → outputs/eda/"
else
  warn "Skipping EDA (--skip-eda)"
fi

# ── Step 4: Agentic Loop ──────────────────────────────────────────────────────
header "Step 4/5 — Agentic Pricing Loop"

STEPS_ARG=""
if [[ -n "$STEPS" ]]; then
  STEPS_ARG="--steps $STEPS"
  info "Running $STEPS steps of the agentic loop..."
else
  info "Running full test set (all steps)..."
fi

$PYTHON orchestrator.py \
  --csv "$UNIFIED" \
  $STEPS_ARG \
  --delay "$DELAY" \
  --verbose 10 \
  --epsilon 0.3 \
  --alpha 4.0 \
  --beta 4.0 \
  --lr 0.8 \
  --decay 0.002 \
  --out outputs/agentic_outcomes.csv \
  --predictions outputs/predictions.csv \
  --log-level INFO

success "Agentic loop complete → outputs/agentic_outcomes.csv"

# ── Step 5: Post-run EDA plots ────────────────────────────────────────────────
header "Step 5/5 — Post-Run Visualisations"

if [[ "$SKIP_EDA" == false ]]; then
  info "Generating post-run plots (predicted vs actual, reward, theta)..."
  $PYTHON -m src.eda.plots \
    --base "$UNIFIED" \
    --acn data/raw/acndata_sessions.json.xlsx \
    --outcomes outputs/agentic_outcomes.csv \
    --predictions outputs/predictions.csv \
    --log-level INFO
  success "Post-run plots saved → outputs/eda/"
else
  warn "Skipping post-run EDA (--skip-eda)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
header "Pipeline Complete"
echo ""
echo -e "  ${GREEN}Outputs:${RESET}"
echo -e "    outputs/agentic_outcomes.csv   — per-step episode log"
echo -e "    outputs/predictions.csv        — actual vs predicted demand"
echo -e "    outputs/sensitivity_analysis.csv — epsilon sweep"
echo -e "    outputs/eda/                   — all EDA and post-run plots"
echo ""
