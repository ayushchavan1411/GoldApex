#!/usr/bin/env bash
# Keep RR1 / pos-RR hunters alive. Memory-aware: skip heavy jobs when RAM tight.
set -u
cd /workspace
TMUX_CFG="/exec-daemon/tmux.portal.conf"
[ -f "$TMUX_CFG" ] || TMUX_CFG=""
tmux_cmd() {
  if [ -n "$TMUX_CFG" ]; then tmux -f "$TMUX_CFG" "$@"; else tmux "$@"; fi
}

ensure_session() {
  local s="$1"
  tmux_cmd has-session -t "=$s" 2>/dev/null || tmux_cmd new-session -d -s "$s" -c /workspace -- "${SHELL:-bash}" -l
}

avail_mb() {
  free -m | awk '/Mem:/{print $7}'
}

start_job() {
  local s="$1"
  local cmd="$2"
  local need="${3:-1800}"
  ensure_session "$s"
  if pgrep -f "python3 -u ${cmd}" >/dev/null 2>&1; then
    return 0
  fi
  local a
  a=$(avail_mb)
  if [ "$a" -lt "$need" ]; then
    echo "$(date -u '+%Y-%m-%d %H:%M:%S') | defer $cmd need=${need}MB avail=${a}MB" >> rr1_search_results/watchdog.log
    return 0
  fi
  echo "$(date -u '+%Y-%m-%d %H:%M:%S') | restart $cmd in $s (avail=${a}MB)" | tee -a rr1_search_results/watchdog.log
  tmux_cmd send-keys -t "${s}:0.0" C-c 2>/dev/null || true
  sleep 1
  tmux_cmd send-keys -t "${s}:0.0" "cd /workspace && python3 -u ${cmd} 2>&1 | tee -a rr1_search_results/${s}.log" C-m
}

mkdir -p rr1_search_results
echo "$(date -u '+%Y-%m-%d %H:%M:%S') | watchdog start (mem-aware)" | tee -a rr1_search_results/watchdog.log

while true; do
  # 1% price RR only (ATR RR removed) — bridge set
  start_job "rr1-pct05" "rr1_pct05_hunt.py" 2200
  start_job "rr1-course" "rr1_course_hunt.py" 2200
  start_job "rr1-posrr" "rr1_posrr_hunt.py" 2200
  start_job "rr1-mohamad" "rr1_mohamad_ict_hunt.py" 2200
  start_job "rr1-freq" "rr1_freqboost.py" 2200
  start_job "rr1-invert" "rr1_invert_search.py" 2000
  {
    echo -n "$(date -u '+%Y-%m-%d %H:%M:%S') | alive: "
    pgrep -af 'python3 -u rr1_' | grep -v pgrep | wc -l
    echo -n " avail_mb="
    avail_mb
  } >> rr1_search_results/watchdog.log
  sleep 120
done
