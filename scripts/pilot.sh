#!/usr/bin/env bash
# SWT-bench pilot: harness -> dataset -> images -> agent + one-shot -> official evaluation -> report.
#
#   conda activate testagent
#   export OPENAI_API_KEY=<the key you gave llama-server>
#   bash scripts/pilot.sh                    # 5 dev instances
#   N=20 WORKERS=2 bash scripts/pilot.sh     # more
#   SUFFIX=v2 bash scripts/pilot.sh          # fresh runs (e.g. after changing the code) next to the old ones
#
# Re-running is safe: existing images are skipped, finished instances are kept, instances that
# ended in an infrastructure error are run again, and evaluation starts fresh.
set -euo pipefail

N=${N:-5}                      # instances
SPLIT=${SPLIT:-dev}
WORKERS=${WORKERS:-2}          # parallel tasks; match llama-server -np
EVAL_WORKERS=${EVAL_WORKERS:-2}
BUILD_WORKERS=${BUILD_WORKERS:-1}  # parallel env builds can run Docker out of memory (exit 137)
LLM_URL=${LLM_URL:-http://localhost:8080/v1}
SWT_DIR=third_party/swt-bench
SUFFIX=${SUFFIX:-}
METHODS=${METHODS:-"agent oneshot"}   # e.g. "agent agent_nofix oneshot" to measure the candidate-fix check
BASE="${SPLIT}${N}"
TAG="${BASE}${SUFFIX:+_$SUFFIX}"

cd "$(dirname "$0")/.."
mkdir -p logs
log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ checks
log "checks"
python -c "import testagent" 2>/dev/null || die "testagent not importable -- conda activate testagent && pip install -e '.[dev]'"
docker info >/dev/null 2>&1 || die "Docker is not usable by this user (try: docker run hello-world)"
# the harness talks to Docker through the Python SDK, which may use a different socket than the CLI
dmsg=$(python -c "from testagent import swt; m = swt.docker_health(); print(m or '')")
[ -z "$dmsg" ] || die "$dmsg"
dmem=$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo 0)
dmem_gb=$(( dmem / 1024 / 1024 / 1024 ))
[ "$dmem_gb" -ge 12 ] || echo "WARNING: Docker sees only ${dmem_gb} GB of memory; environment builds may be killed (137)"
[ -n "${OPENAI_API_KEY:-}" ] || die "OPENAI_API_KEY is empty -- export the key you gave llama-server"
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $OPENAI_API_KEY" "$LLM_URL/models" || true)
[ "$code" = "200" ] || die "LLM server at $LLM_URL answered HTTP $code (is llama-server running? right key?)"
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
need=$(( N * 3 + 20 ))          # rough: instance images share their environment images
[ "$free_gb" -ge "$need" ] || echo "WARNING: ${free_gb} GB free here; ~${need} GB may be needed for $N images (Docker's own disk limit counts too)"
echo "python $(python -V 2>&1 | cut -d' ' -f2), docker ok (CLI + SDK, ${dmem_gb} GB memory), LLM ok, ${free_gb} GB free"

# ----------------------------------------------------------------- harness
if [ ! -f "$SWT_DIR/src/main.py" ]; then
  log "make swt (clone + install the official harness)"
  make swt 2>&1 | tee logs/swt_install.log
else
  log "harness present: $SWT_DIR"
fi

# ----------------------------------------------------------------- dataset
if [ ! -s data/swt_lite.jsonl ]; then
  log "make prepare (download SWT-bench Lite, dev/test split)"
  make prepare 2>&1 | tee logs/prepare.log
else
  log "dataset present: data/swt_lite.jsonl ($(wc -l < data/swt_lite.jsonl) instances)"
fi

# ------------------------------------------------------------------ images
log "building images for the first $N $SPLIT instances (slow the first time)"
python bench/build_images.py --split "$SPLIT" --limit "$N" --workers "$BUILD_WORKERS" 2>&1 | tee "logs/images_${TAG}.log" \
  || die "no image could be built -- see logs/images_${TAG}.log"
# instances whose image failed are skipped by run.py (listed in its output) and can be added later

# ------------------------------------------------------- harness check (gold)
# the dataset's own reference tests must resolve; an instance where they do not is unsolvable here
GOLD="gold_${BASE}"
if [ ! -s "runs/$GOLD/eval.json" ]; then
  log "harness check: evaluate the reference tests ($GOLD)"
  python bench/gold.py --split "$SPLIT" --limit "$N" --run "$GOLD" 2>&1 | tee "logs/gold_${BASE}.log"
  python bench/evaluate.py --run "$GOLD" --workers "$EVAL_WORKERS" 2>&1 | tee -a "logs/gold_${BASE}.log"
else
  log "harness check present: runs/$GOLD/eval.json"
fi

# -------------------------------------------------------------------- runs
for METHOD in $METHODS; do
  RUN="${METHOD}_${TAG}"
  log "run $RUN"
  python bench/run.py --method "$METHOD" --split "$SPLIT" --limit "$N" --workers "$WORKERS" \
      --run "$RUN" --set "llm.base_url=$LLM_URL" 2>&1 | tee "logs/run_${RUN}.log"
  log "evaluate $RUN (official harness)"
  python bench/evaluate.py --run "$RUN" --workers "$EVAL_WORKERS" 2>&1 | tee "logs/eval_${RUN}.log"
done

# ------------------------------------------------------------------ report
log "report"
RUNS=$(for m in $METHODS; do printf '%s_%s ' "$m" "$TAG"; done)
python bench/report.py $RUNS --ref "oneshot_${TAG}" --gold "$GOLD" 2>&1 | tee "logs/report_${TAG}.log"

log "done -- send me the archive below (runs, logs and the harness's per-instance logs)"
tar czf "pilot_${TAG}.tgz" $(for m in $METHODS; do printf 'runs/%s_%s ' "$m" "$TAG"; done) "runs/$GOLD" \
    runs/report.md logs -C "$SWT_DIR" \
    $(for m in $METHODS; do printf 'run_instance_swt_logs/%s_%s ' "$m" "$TAG"; done) \
    "run_instance_swt_logs/$GOLD" 2>/dev/null || true
echo "  $(pwd)/pilot_${TAG}.tgz"
