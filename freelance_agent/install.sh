#!/usr/bin/env bash
# Установка фриланс-агента на Ubuntu одной командой:  ./install.sh
# Повторный запуск обновляет агента (после git pull) и перезапускает службу.
#   FA_SKIP_SETUP=1  — не запускать мастер настройки
#   FA_NO_SERVICE=1  — не ставить автозапуск (systemd)
#   FREELANCE_AGENT_HOME — где хранить настройки и данные (по умолчанию ~/freelance-agent)
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_HOME="${FREELANCE_AGENT_HOME:-$HOME/freelance-agent}"
VENV="$APP_HOME/venv"
BIN="$VENV/bin/freelance-agent"
CONFIG="$APP_HOME/config.yaml"

say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  command -v sudo >/dev/null || die "нужен sudo для установки пакетов"
  SUDO="sudo"
fi

# --- 1. системные пакеты -----------------------------------------------------------------
if [ -r /etc/os-release ]; then . /etc/os-release; fi
case "${ID:-} ${ID_LIKE:-}" in
  *ubuntu*|*debian*) ;;
  *) warn "скрипт рассчитан на Ubuntu/Debian, у вас ${PRETTY_NAME:-неизвестная ОС}" ;;
esac

say "Ставлю системные пакеты (python, bubblewrap, socat)"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq python3 python3-venv python3-pip bubblewrap socat ca-certificates >/dev/null

PY=python3
if ! $PY -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
  say "Системный Python старше 3.11, ставлю python3.11"
  $SUDO apt-get install -y -qq python3.11 python3.11-venv >/dev/null || die "не удалось поставить python3.11"
  PY=python3.11
fi

# --- 2. сам агент ------------------------------------------------------------------------
say "Устанавливаю агента в $VENV"
mkdir -p "$APP_HOME"
[ -x "$VENV/bin/python" ] || $PY -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "$SRC"

mkdir -p "$HOME/.local/bin"
ln -sf "$BIN" "$HOME/.local/bin/freelance-agent"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) warn "добавьте ~/.local/bin в PATH (или перелогиньтесь): export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac

# --- 3. песочница для выполнения заказов --------------------------------------------------
probe() {
  "$VENV/bin/python" -c 'import sys; from freelance_agent.agents.executor import sandbox_problem as p
r = p(); print(r or "ok"); sys.exit(1 if r else 0)'
}
if probe >/dev/null; then
  say "Песочница работает"
else
  restrict="$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)"
  BWRAP="$(command -v bwrap)"
  if [ "$restrict" = "1" ] && ! grep -lsq "$BWRAP" /etc/apparmor.d/* 2>/dev/null; then
    # Ubuntu 23.10+ запрещает user namespaces без профиля AppArmor: разрешаем только bubblewrap.
    say "Разрешаю bubblewrap создавать namespaces (профиль AppArmor)"
    $SUDO tee /etc/apparmor.d/freelance-agent-bwrap >/dev/null <<EOF
abi <abi/4.0>,
include <tunables/global>

profile freelance-agent-bwrap $BWRAP flags=(unconfined) {
  userns,

  include if exists <local/freelance-agent-bwrap>
}
EOF
    $SUDO apparmor_parser -r /etc/apparmor.d/freelance-agent-bwrap
  fi
  if probe >/dev/null; then
    say "Песочница работает"
  else
    warn "песочница не запускается: $(probe || true)"
    warn "поиск заказов и отклики будут работать, но выполнять заказы агент откажется"
  fi
fi

# --- 4. настройка ------------------------------------------------------------------------
if [ -f "$CONFIG" ]; then
  say "Настройки уже есть: $CONFIG (заново: freelance-agent setup)"
elif [ "${FA_SKIP_SETUP:-0}" = "1" ]; then
  "$BIN" -c "$CONFIG" init >/dev/null
  say "Создан пример настроек: $CONFIG"
else
  say "Мастер настройки"
  "$BIN" -c "$CONFIG" setup
fi

# --- 5. автозапуск -----------------------------------------------------------------------
UNIT_BODY="[Unit]
Description=Freelance Agent (Claude)
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=$APP_HOME
ExecStart=$BIN -c $CONFIG run
Restart=on-failure
RestartSec=60
Environment=PYTHONUNBUFFERED=1"

ready() { grep -qE '^ANTHROPIC_API_KEY=.+' "$APP_HOME/.env" 2>/dev/null; }

if [ "${FA_NO_SERVICE:-0}" = "1" ] || [ ! -d /run/systemd/system ]; then
  warn "автозапуск не настроен (нет systemd). Запуск вручную: freelance-agent run"
elif [ "$(id -u)" -eq 0 ]; then
  printf '%s\nUser=root\n\n[Install]\nWantedBy=multi-user.target\n' "$UNIT_BODY" > /etc/systemd/system/freelance-agent.service
  systemctl daemon-reload
  systemctl enable freelance-agent >/dev/null 2>&1
  if ready; then systemctl restart freelance-agent; fi
  LOGS="journalctl -u freelance-agent -f"
  CTL="systemctl"
else
  mkdir -p "$HOME/.config/systemd/user"
  printf '%s\n\n[Install]\nWantedBy=default.target\n' "$UNIT_BODY" > "$HOME/.config/systemd/user/freelance-agent.service"
  systemctl --user daemon-reload
  systemctl --user enable freelance-agent >/dev/null 2>&1
  # работать и после выхода из системы / перезагрузки
  loginctl enable-linger "$USER" 2>/dev/null || $SUDO loginctl enable-linger "$USER" || true
  if ready; then systemctl --user restart freelance-agent; fi
  LOGS="journalctl --user -u freelance-agent -f"
  CTL="systemctl --user"
fi

echo
say "Готово."
echo "  Проверка:          freelance-agent check"
echo "  Обзор рынка (0 ₽): freelance-agent scan"
echo "  Отчёт:             freelance-agent stats   (или /stats в боте)"
if [ -n "${CTL:-}" ]; then
  if ready; then
    echo "  Агент запущен как служба. Логи: $LOGS"
  else
    echo "  Впишите ANTHROPIC_API_KEY в $APP_HOME/.env и запустите: $CTL start freelance-agent"
  fi
  echo "  Стоп/старт:        $CTL stop|start freelance-agent"
fi
