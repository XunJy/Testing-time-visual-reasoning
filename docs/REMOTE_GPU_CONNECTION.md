# Remote GPU connection handoff

This is the reproducible procedure for connecting a fresh Google Colab GPU
runtime to Xunjie's Mac and the Codex desktop app through Tailscale SSH. Repeat
the runtime-side steps after every Colab reset.

## Rules that prevent most failures

1. Never reuse an IP or Tailscale login URL from an old runtime.
2. The Mac and Colab node must use the same Tailscale account.
3. Use the non-root Linux user `codex` for SSH and Codex authentication.
4. Verify Tailscale, SSH, and the GPU separately before running an experiment.
5. Save results back to the Mac; the Colab VM is temporary.

## 1. Verify the GPU in Colab

Select **Runtime > Change runtime type**, choose a GPU, and run:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Continue only if a real GPU name and its memory are printed.

## 2. Start Tailscale SSH in Colab

Run this in one cell. The `tailscale up` process is put in the background so
the cell prints the login URL instead of appearing to hang for minutes.

```bash
%%bash
set -euo pipefail

TS_DIR=/tmp/tailscale
SOCKET="$TS_DIR/tailscaled.sock"
DAEMON_LOG=/tmp/tailscaled.log
UP_LOG=/tmp/tailscale-up.log
mkdir -p "$TS_DIR"

if ! command -v tailscale >/dev/null 2>&1; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi

id codex >/dev/null 2>&1 || useradd -m -s /bin/bash codex

if pgrep -x tailscaled >/dev/null 2>&1; then
  pkill -x tailscaled || true
  for _ in $(seq 1 20); do
    pgrep -x tailscaled >/dev/null 2>&1 || break
    sleep 0.25
  done
fi

rm -f "$SOCKET"
nohup "$(command -v tailscaled)" \
  --tun=userspace-networking \
  --socket="$SOCKET" \
  --state="$TS_DIR/tailscaled.state" \
  --statedir="$TS_DIR" \
  >"$DAEMON_LOG" 2>&1 </dev/null &

for _ in $(seq 1 40); do
  [ -S "$SOCKET" ] && break
  sleep 0.25
done

if [ ! -S "$SOCKET" ]; then
  echo "ERROR: tailscaled did not create its socket"
  tail -80 "$DAEMON_LOG" || true
  exit 1
fi

HOST="colab-codex-$(date +%m%d-%H%M)"
: >"$UP_LOG"
nohup "$(command -v tailscale)" --socket="$SOCKET" up \
  --ssh --hostname="$HOST" \
  >"$UP_LOG" 2>&1 </dev/null &

for _ in $(seq 1 40); do
  grep -Eq 'https://login.tailscale.com|Success|already logged in' "$UP_LOG" \
    2>/dev/null && break
  sleep 0.25
done

cat "$UP_LOG"
echo
echo "Open the NEW login URL above, then run the status cell."
```

Open the newly printed Tailscale URL and authenticate. Do not save that URL in
the repository.

## 3. Obtain the current Tailscale IP

After authentication, run:

```bash
%%bash
set -euo pipefail
SOCKET=/tmp/tailscale/tailscaled.sock

if [ ! -S "$SOCKET" ]; then
  echo "ERROR: tailscaled is not running; rerun the setup cell."
  tail -80 /tmp/tailscaled.log 2>/dev/null || true
  exit 1
fi

IP=""
for _ in $(seq 1 30); do
  IP=$(tailscale --socket="$SOCKET" ip -4 2>/dev/null || true)
  [ -n "$IP" ] && break
  sleep 1
done

echo "=== STATUS ==="
timeout 10 tailscale --socket="$SOCKET" status || true
echo "=== IP ==="
echo "TAILSCALE_IP=$IP"
test -n "$IP"
```

Use only the newly printed `TAILSCALE_IP` for this runtime.

## 4. Prepare the remote shell and install Codex

Run once per new runtime:

```bash
%%bash
set -euo pipefail

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  curl git rsync tmux xz-utils

id codex >/dev/null 2>&1 || useradd -m -s /bin/bash codex
mkdir -p /home/codex/project
chown -R codex:codex /home/codex

GPU_ENV='export LD_LIBRARY_PATH=/usr/lib64-nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'
grep -qxF "$GPU_ENV" /home/codex/.bashrc 2>/dev/null \
  || printf '%s\n' "$GPU_ENV" >>/home/codex/.bashrc

if [ -d /usr/lib64-nvidia ]; then
  printf '%s\n' /usr/lib64-nvidia >/etc/ld.so.conf.d/colab-nvidia.conf
  ldconfig
fi

case "$(uname -m)" in
  x86_64) NODE_ARCH=x64 ;;
  aarch64|arm64) NODE_ARCH=arm64 ;;
  *) echo "Unsupported architecture: $(uname -m)"; exit 1 ;;
esac

NODE_VERSION=$(curl -fsSL https://nodejs.org/dist/index.tab \
  | awk 'NR > 1 && $10 != "-" && !found {print $1; found=1}')
test -n "$NODE_VERSION"

rm -rf /opt/node-lts
mkdir -p /opt/node-lts
curl -fsSL "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" \
  | tar -xJ --strip-components=1 -C /opt/node-lts

ln -sfn /opt/node-lts/bin/node /usr/local/bin/node
ln -sfn /opt/node-lts/bin/npm /usr/local/bin/npm
ln -sfn /opt/node-lts/bin/npx /usr/local/bin/npx
npm install -g @openai/codex
ln -sfn /opt/node-lts/bin/codex /usr/local/bin/codex

echo "NODE=$(node --version)"
echo "NPM=$(npm --version)"
echo "CODEX=$(codex --version)"
```

Do not authenticate Codex as root. Authentication is done later through SSH as
the `codex` user.

## 5. Configure and test SSH on the Mac

The Tailscale Mac app must show **Connected** under the same account. Replace
the example IP and test the private connection:

```bash
IP=100.x.y.z
TSCLI=/Applications/Tailscale.app/Contents/MacOS/Tailscale
TAILSCALE_BE_CLI=1 "$TSCLI" ping --c 3 "$IP"
```

`via DERP(...)` is acceptable. It means the traffic is relayed rather than
direct.

Add or update one concrete entry in `~/.ssh/config`:

```sshconfig
Host colab-codex-current
  HostName 100.x.y.z
  User codex
  ConnectTimeout 20
  ServerAliveInterval 15
  ServerAliveCountMax 3
  StrictHostKeyChecking accept-new
```

Replace only `HostName` when the runtime changes. Test SSH and the GPU:

```bash
ssh colab-codex-current \
  'echo SSH_OK; whoami; hostname; LD_LIBRARY_PATH=/usr/lib64-nvidia nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

The expected output contains `SSH_OK`, `codex`, a hostname, and a GPU name.
The output itself is not another command to paste into Terminal.

## 6. Authenticate remote Codex and connect the desktop app

Enter the remote shell:

```bash
ssh colab-codex-current
```

Then run remotely:

```bash
codex --version
codex login --device-auth
codex login status
```

Device authentication is appropriate for a headless runtime. Never copy the
Mac's `~/.codex/auth.json`; it contains credentials.

In the Codex desktop app:

1. Open **Settings > Connections**.
2. Add or enable `colab-codex-current`.
3. Select `/home/codex/project` as the remote project directory.

The app discovers concrete hosts from `~/.ssh/config` and starts its remote
service through the remote `codex` executable.

## 7. Synchronize the project safely

From the Mac:

```bash
LOCAL='/Users/xunj/Desktop/Testing-Time Visual Reasoning'
REMOTE=colab-codex-current

ssh "$REMOTE" 'mkdir -p /home/codex/project'
rsync -az \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='Old Patch/' \
  --exclude='remote_returns/' \
  --exclude='scratch/' \
  --exclude='tmp/' \
  "$LOCAL/" "$REMOTE:/home/codex/project/"
```

`Old Patch/` is intentionally excluded: the new FuDD line has no runtime
dependency on the roughly 753 MB historical tree. Never add `--delete`. Before
the runtime ends, return results into a new local
directory:

```bash
RUN_ID=$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOCAL/remote_returns/$RUN_ID"
rsync -az "$REMOTE:/home/codex/project/experiments/" \
  "$LOCAL/remote_returns/$RUN_ID/experiments/"
```

For long jobs, run `tmux new -s reasoning`. Detach with `Ctrl-b`, then `d`, and
reattach with `tmux attach -t reasoning`. This survives an SSH disconnect but
not deletion of the Colab VM.

## Troubleshooting

If a Colab cell appears stuck, interrupt it and run:

```bash
%%bash
SOCKET=/tmp/tailscale/tailscaled.sock
echo "=== PROCESSES ==="
pgrep -a tailscaled || true
echo "=== SOCKET ==="
ls -l "$SOCKET" || true
echo "=== STATUS ==="
timeout 10 tailscale --socket="$SOCKET" status || true
echo "=== LOGS ==="
tail -80 /tmp/tailscaled.log 2>/dev/null || true
tail -80 /tmp/tailscale-up.log 2>/dev/null || true
```

- Missing socket or daemon: rerun the Tailscale setup cell.
- `NeedsLogin` or `Logged out`: obtain and open a new login URL.
- Ping works but SSH times out: in Colab run
  `tailscale --socket=/tmp/tailscale/tailscaled.sock set --ssh`, check the
  tailnet SSH policy, and confirm that the `codex` user exists.
- `nvidia-smi` cannot find `libnvidia-ml.so`: run it with
  `LD_LIBRARY_PATH=/usr/lib64-nvidia` and rerun the shell preparation cell.
- Changed SSH host key: run `ssh-keygen -R colab-codex-current` and
  `ssh-keygen -R 100.x.y.z`, then reconnect with `accept-new`.
- Offline old runtime: it cannot be revived through Tailscale; start a new
  runtime and repeat the procedure.

## Security and lifetime

- Never commit login URLs, auth keys, API keys, or Codex auth files.
- Colab VMs are deleted after idle periods and have a maximum lifetime. Paid
  compute improves availability but does not make the VM persistent.
- Colab's FAQ restricts remote-control or SSH activity on the free tier without
  a positive compute-unit balance. Confirm the current policy before relying
  on this workflow.

## Official references

- [OpenAI: Remote connections and SSH hosts](https://learn.chatgpt.com/docs/remote-connections)
- [OpenAI: Codex authentication](https://learn.chatgpt.com/docs/auth)
- [OpenAI: Codex CLI installation](https://help.openai.com/en/articles/11096431)
- [Tailscale: Userspace networking](https://tailscale.com/docs/concepts/userspace-networking)
- [Tailscale: Tailscale SSH](https://tailscale.com/docs/features/tailscale-ssh)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
