#!/usr/bin/env bash
# RunPod container entrypoint.
#
# Designed to never exit, even if a step here fails, so the container keeps
# running and you can debug via the RunPod web terminal.
#
# - Generates SSH host keys if the image doesn't already have them.
# - Writes the RunPod-injected PUBLIC_KEY into /root/.ssh/authorized_keys so
#   you can `ssh root@<host> -p <port>` and `scp`/`rsync` from your laptop.
# - Starts sshd in the background.
# - Sleeps forever to keep PID 1 alive.

mkdir -p /root/.ssh
chmod 700 /root/.ssh

if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# Custom Docker images often ship without SSH host keys; ssh-keygen -A is
# idempotent (only generates the keys that don't already exist).
ssh-keygen -A

mkdir -p /var/run/sshd

# Don't `set -e` above this — if sshd fails to start for any reason we still
# want the container alive so you can inspect logs from the web terminal.
/usr/sbin/sshd || echo "WARNING: sshd failed to start; check /etc/ssh/sshd_config and 'cat /var/log/auth.log'"

exec sleep infinity
