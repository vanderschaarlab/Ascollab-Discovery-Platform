# Sandbox

Agents run their analysis code in Docker containers managed by a small server,
`sandbox/server_docker.py`. Each research session gets its own container with a persistent Python
namespace, like a notebook kernel.

## Build the image

```bash
docker build -t agent-sandbox:py311 \
  --build-arg SANDBOX_UID=$(id -u) --build-arg SANDBOX_GID=$(id -g) sandbox/
```

The image installs `sandbox/server_requirements.txt`: the scientific packages agents' code can
import (`env_setup/packages.py` lists the ones agents are told about). The container runs as the
user given by `SANDBOX_USER_UID` / `SANDBOX_USER_GID`; build with the IDs of the user who runs the
server and set those two variables to the same values, so the container can write the session
folders the server creates.

## Run the server

```bash
export SANDBOX_IMAGE=agent-sandbox:py311
export SANDBOX_DATASETS_DIR=$PWD/sandbox/datasets
export SANDBOX_USER_UID=$(id -u) SANDBOX_USER_GID=$(id -g)
gunicorn sandbox.server_docker:app -k gthread -w 1 --threads 8 \
  -b 127.0.0.1:9000 --timeout 180 --graceful-timeout 30 --keep-alive 30
```

The pipeline finds the server at `SANDBOX_BASE_URL` (default `http://127.0.0.1:9000`).

The server has no authentication: anyone who can reach it can run code on your machine. Bind it
to `127.0.0.1` (as above) and do not expose its port.

## Settings

All are environment variables read by the server (`SANDBOX_` prefix; a `.env` file also works).

| Variable | Default | Meaning |
|---|---|---|
| `SANDBOX_IMAGE` | `agent-sandbox:py311` | Image for session containers |
| `SANDBOX_DATASETS_DIR` | `sandbox/datasets` | Mounted read-only at `/datasets` (see [data.md](data.md)) |
| `SANDBOX_SESSIONS_ROOT` | `/tmp/sandbox_sessions` | Host folder for each session's `/work` |
| `SANDBOX_MAX_SESSIONS` | 40 | Concurrent sessions; the pipeline sizes its admission gate from it |
| `SANDBOX_CPUS` | 2.0 | CPUs per container |
| `SANDBOX_MEMORY` | `6g` | Memory per container |
| `SANDBOX_PIDS_LIMIT` | 256 | Processes per container |
| `SANDBOX_NETWORK_DISABLED` | true | No network inside containers |
| `SANDBOX_READ_ONLY_ROOTFS` | true | Read-only root filesystem (`/work` and `/tmp` are writable) |
| `SANDBOX_NO_NEW_PRIVILEGES` | true | `no-new-privileges`; all capabilities are dropped |
| `SANDBOX_USER_UID`, `SANDBOX_USER_GID` | 1000, 999 | User the container runs as |
| `SANDBOX_DEFAULT_TIMEOUT` | 120 | Seconds per code cell when the caller gives none |
| `SANDBOX_INSTALL_BUDGET_S` | 180 | Extra seconds for installing missing packages (only with network on) |
| `SANDBOX_MAX_STATE_BYTES` | 268435456 (256 MB) | Largest variable kept between cells |
| `SANDBOX_RUN_ID` | today's date | Groups session folders under `SANDBOX_SESSIONS_ROOT` |

## State between cells

After each cell that runs without error the runner saves the session's variables so a restarted
container can continue; a cell that fails or times out saves nothing.
Each variable is saved only if it pickles within `SANDBOX_MAX_STATE_BYTES`, and only if the save
reaches it in time: the save stops about 1.5 s before the server would stop waiting for the cell
(timeout + install allowance + 3 s). Larger, unpicklable or unreached variables stay in memory but
are not saved, and the cell's history entry lists them under `state_skipped`. With the network
off, missing packages are not installed; agents work with the packages in the image.
