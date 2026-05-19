#!/usr/bin/env bash
# RunPod container entrypoint.
#
# - Installs the SSH public key RunPod injects via the PUBLIC_KEY env var so
#   `ssh root@<pod-host> -p <pod-port>` and `scp/rsync` work from your laptop.
# - Starts sshd in the background.
# - Sleeps forever so the container stays alive; you connect via the web
#   terminal or SSH and run training from there.

set -e

mkdir -p /root/.ssh
chmod 700 /root/.ssh

if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# sshd refuses to start without this directory
mkdir -p /var/run/sshd

/usr/sbin/sshd

exec sleep infinity
