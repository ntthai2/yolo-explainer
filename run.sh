#!/usr/bin/env bash
set -e

PORT=${1:-8008}
PYTHON_EXEC="/srv/nns/envs/nda_cv/bin/python"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${APP_DIR}:${PYTHONPATH}"
cd "${APP_DIR}"

# --- Cấu hình Tunnel qua Jump Host (10.2.3.62) ---
ENABLE_TUNNEL=${ENABLE_TUNNEL:-true}
JUMP_HOST="10.2.3.62"
JUMP_USER="cmc"
SSH_KEY="/srv/nns/users/ntthai/.ssh/id_ed25519"
REMOTE_BIND=$((20000 + PORT))
PUBLIC_PORT="${PORT}"

if [ "$ENABLE_TUNNEL" = "true" ] && [ -f "$SSH_KEY" ]; then
    echo "=== Thiết lập đường hầm mạng LAN qua ${JUMP_HOST} ==="

    # 1. SSH Reverse Tunnel
    if ! pgrep -f "ssh.*-R ${REMOTE_BIND}:127.0.0.1:${PORT}" > /dev/null; then
        echo "[1/2] Đang mở SSH Reverse Tunnel (127.0.0.1:${PORT} -> ${JUMP_HOST}:${REMOTE_BIND})..."
        ssh -o StrictHostKeyChecking=no -f -N -i "$SSH_KEY" -R ${REMOTE_BIND}:127.0.0.1:${PORT} ${JUMP_USER}@${JUMP_HOST} || echo "Cảnh báo: Không thể tạo SSH tunnel"
    else
        echo "[1/2] SSH Reverse Tunnel đã sẵn sàng."
    fi

    # 2. Socat trên Jump host
    echo "[2/2] Đang kiểm tra socat trên ${JUMP_HOST}:${PUBLIC_PORT}..."
    ssh -o BatchMode=yes -o ConnectTimeout=3 -i "$SSH_KEY" ${JUMP_USER}@${JUMP_HOST} "
        if ! ss -tlpn | grep -q \":${PUBLIC_PORT} \"; then
            nohup /usr/bin/socat TCP-LISTEN:${PUBLIC_PORT},fork,reuseaddr TCP:127.0.0.1:${REMOTE_BIND} > /dev/null 2>&1 &
            sleep 0.5
            if ss -tlpn | grep -q \":${PUBLIC_PORT} \"; then
                echo 'Đã bật socat thành công.'
            else
                echo 'Cảnh báo: Chưa bật được socat trên jump host.'
            fi
        else
            echo 'Socat đã chạy sẵn.'
        fi
    " || echo "Cảnh báo: Không thể kết nối tới jump host ${JUMP_HOST}"
    echo "------------------------------------------------"
fi

echo "Khởi động YOLO Explainer trên cổng ${PORT}..."
echo "👉 Local:    http://localhost:${PORT}"
if [ "$ENABLE_TUNNEL" = "true" ]; then
    echo "👉 Mạng LAN: http://${JUMP_HOST}:${PUBLIC_PORT}"
fi
echo "------------------------------------------------"

exec "${PYTHON_EXEC}" -m uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT}"
