#!/usr/bin/env bash
# Start, restart, stop or check the Bajurinda simulator.
#   bajurinda            start (restarts if already running)
#   bajurinda stop       stop it
#   bajurinda status     say whether it is running
set -u
PORT=8501
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

running_pids() {
  lsof -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null
}

stop_app() {
  local pids
  pids="$(running_pids)"
  if [ -z "$pids" ]; then
    echo "Nothing is running on port ${PORT}."
    return 0
  fi
  echo "Stopping the app on port ${PORT} (pid ${pids//$'\n'/, })..."
  kill $pids 2>/dev/null
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    [ -z "$(running_pids)" ] && return 0
    sleep 0.3
  done
  kill -9 $pids 2>/dev/null
}

case "${1:-start}" in
  stop)
    stop_app
    ;;
  status)
    if [ -n "$(running_pids)" ]; then
      echo "Running at http://localhost:${PORT}"
    else
      echo "Not running."
    fi
    ;;
  start|restart)
    if [ ! -x "$DIR/.venv/bin/streamlit" ]; then
      echo "Can't find $DIR/.venv/bin/streamlit. Create the environment first (see README)." >&2
      exit 1
    fi
    stop_app
    echo "Starting the Bajurinda simulator at http://localhost:${PORT} (Ctrl+C here to stop it)"
    cd "$DIR" && exec "$DIR/.venv/bin/streamlit" run app.py --server.port "$PORT"
    ;;
  *)
    echo "Usage: bajurinda [start|restart|stop|status]" >&2
    exit 2
    ;;
esac
