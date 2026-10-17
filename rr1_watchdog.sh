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
  # Priority: new gates (pos-RR) + OF/ORB/goal/lowwr
  start_job "rr1-posrr" "rr1_posrr_hunt.py" 2200
  start_job "rr1-mohamad" "rr1_mohamad_ict_hunt.py" 2200
  start_job "rr1-orderflow" "rr1_orderflow_search.py" 2000
  start_job "rr1-goal" "rr1_goal_allsession.py" 2000
  start_job "rr1-orbwf" "rr1_orb_walkforward.py" 2000
  start_job "rr1-lowwr" "rr1_lowwr_invert.py" 2000
  start_job "rr1-invert" "rr1_invert_search.py" 2000
  # Mid priority
  start_job "rr1-select" "rr1_select_search.py" 2200
  start_job "rr1-public" "rr1_public_search.py" 2200
  start_job "rr1-freq" "rr1_freqboost.py" 2200
  start_job "rr1-regime" "rr1_regime_search.py" 2200
  # Heavy — only when plenty of RAM
  start_job "rr1-union" "rr1_union_search.py" 3500
  start_job "rr1-pipeline" "rr1_pipeline_search.py" 3500
  start_job "rr1-ictstack" "rr1_ict_stack_search.py" 3500
  start_job "rr1-sllearn" "rr1_sl_learn_search.py" 3500
  start_job "rr1-allcombo" "rr1_allcombo_search.py" 4000
  start_job "rr1-orcombo" "rr1_orcombo.py" 4000
  start_job "rr1-hill" "rr1_hillclimb.py" 4000
  start_job "rr1-smc" "rr1_smc_search.py" 4000
  start_job "rr1-random" "rr1_random_opt.py" 4000
  start_job "rr1-search" "rr1_search.py" 4500
  {
    echo -n "$(date -u '+%Y-%m-%d %H:%M:%S') | alive: "
    pgrep -af 'python3 -u rr1_' | grep -v pgrep | wc -l
    echo -n " avail_mb="
    avail_mb
  } >> rr1_search_results/watchdog.log
  sleep 120
done
