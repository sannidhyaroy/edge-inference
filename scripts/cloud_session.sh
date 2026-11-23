#!/usr/bin/env bash
# Measure the cloud tier on a Colab T4, unattended, from this machine.
#
#   TAILCAT=/path/to/tailcat scripts/cloud_session.sh
#
# One sitting, about 45 minutes of VM time:
#
#  1. Ask Colab for a T4 until one is in WANT_REGION (default Singapore), up
#     to TRIES (default 5) sessions; then keep whichever region came last.
#     Colab does not let the region be chosen, and a real deployment would
#     pick the nearest one.
#  2. Send the code at HEAD (not a clone: unpushed commits must be there too)
#     and the device's photo hashes. The VM encodes the test photos itself
#     and stops unless every one matches.
#  3. Time the cloud models on the GPU, and bring the tables back at once.
#  4. Start the menu server and `tailcat serve` on the VM; forward a local
#     port to it through tailcat, which goes direct (UDP hole punching,
#     WireGuard inside) when it can, and record whether it did.
#  5. Record the link twice for 10 minutes, one to tune on, one to test on.
#  6. Run local, offload and hedged plans live against the cloud model.
#  7. Stop everything; the VM goes back to Colab.
#
# DRY_RUN=1 rehearses every step on a free VM without a GPU: the models run
# in PyTorch on its CPU, on a few photos, with short probes and few frames,
# and everything it measures stays in logs/, never in results/.
#
# Control goes through `colab exec` (the notebook kernel), never SSH, so
# nothing waits on a background process. Each step's output goes to the log.
# The tailcat log names the VM's public address, so it stays in logs/, which
# git ignores.
set -euo pipefail

SESSION=edge-t4
WANT_REGION=${WANT_REGION:-Singapore}
TRIES=${TRIES:-5}
COLAB=${COLAB:-$HOME/.local/bin/colab}
TAILCAT=${TAILCAT:?set TAILCAT to the tailcat binary}
LOCAL_PORT=18765
DEVICE_VARIANT=mnv4m@160/float32
CLOUD_MODEL=convnextv2_b@224/float32
SERVER_FILES=(
    "convnextv2_b@224_float32" "convnext_t@224_float32"
)
if [[ -n ${DRY_RUN:-} ]]; then
    accelerator=() backend=torch-cpu images=5 probe_seconds=20 frames=6 TRIES=1
else
    accelerator=(--gpu T4) backend=torch-cuda images=1000 probe_seconds=600 frames=200
fi

cd "$(dirname "$0")/.."
mkdir -p logs
stamp=$(date +%Y%m%d-%H%M%S)
log=logs/cloud-session-$stamp.log
exec > >(tee -a "$log") 2>&1
work=$(mktemp -d)
if [[ -n ${DRY_RUN:-} ]]; then
    out=logs/cloud-dryrun-$stamp
    mkdir -p "$out"
else
    out=results
fi
forward_pid=""

cleanup() {
    local status=$?
    echo "== cleanup (exit $status) $(date +%T)"
    if [[ -n $forward_pid ]]; then
        kill "$forward_pid" 2>/dev/null || true
    fi
    "$COLAB" stop -s "$SESSION" >/dev/null 2>&1 || true
    "$COLAB" sessions || true
    cp "$work"/*.out "$work"/*.log "logs/" 2>/dev/null || true
    rm -rf "$work"
    echo "== finished with exit $status at $(date +%T); log $log"
}
trap cleanup EXIT

# One step on the VM; it succeeded only if it printed STEP OK.
vm() {
    local step=$1
    echo "== vm $step $(date +%T)"
    "$COLAB" exec -s "$SESSION" -f scripts/cloud_vm.py --env "STEP=$step" \
        --env "BACKEND=$backend" --env "IMAGES=$images" --timeout 1200 \
        | tee "$work/cloud-$stamp-$step.out"
    grep -q '^STEP OK' "$work/cloud-$stamp-$step.out"
}

echo "== code at $(git rev-parse --short HEAD)"
if [[ -n $(git status --porcelain -- src pyproject.toml uv.lock scripts) ]]; then
    echo "uncommitted changes to the code: commit first, since the VM gets HEAD" >&2
    exit 1
fi
git archive --format=tar.gz -o "$work/code.tgz" HEAD -- \
    src pyproject.toml uv.lock .python-version manifests README.md

region=""
for try in $(seq 1 "$TRIES"); do
    echo "== try $try of $TRIES: a T4 session $(date +%T)"
    "$COLAB" stop -s "$SESSION" >/dev/null 2>&1 || true
    if ! "$COLAB" new -s "$SESSION" "${accelerator[@]}"; then
        echo "no session could be created" >&2
        exit 3
    fi
    vm where
    region=$(sed -n 's/^REGION //p' "$work/cloud-$stamp-where.out")
    echo "== region: $region"
    [[ $region == "$WANT_REGION" ]] && break
done
link=cloud-t4-$(tr '[:upper:] ' '[:lower:]-' <<<"$region")
echo "== link name: $link"

"$COLAB" upload -s "$SESSION" "$work/code.tgz" /content/code.tgz
for photo in jpeg256q85 jpeg160q85; do
    "$COLAB" upload -s "$SESSION" "data/payloads/$photo.csv" "/content/expected-$photo.csv"
done
vm setup
vm profile
for model in "${SERVER_FILES[@]}"; do
    for photo in jpeg256q85 jpeg160q85; do
        name=menu_server_cloud-t4_${model}_${photo}.csv.gz
        "$COLAB" download -s "$SESSION" "/content/edge/results/$name" "$out/$name"
    done
done

vm start
address=$(sed -n 's/^ADDRESS //p' "$work/cloud-$stamp-start.out")
echo "== tailcat address found: ${address:+yes}"
"$TAILCAT" forward "$address" "$LOCAL_PORT:8765" >"$work/cloud-$stamp-forward.log" 2>&1 &
forward_pid=$!
if "$TAILCAT" ping --until-direct --timeout=60s "$address" >"$work/cloud-$stamp-ping.log" 2>&1; then
    path=direct
else
    path=relayed
fi
echo "== tailcat path: $path"
sleep 5

server=127.0.0.1:$LOCAL_PORT
for label in tune test; do
    echo "== probe $label $(date +%T)"
    uv run edge probe --server "$server" --link "$link" --label "$label" \
        --seconds "$probe_seconds" --out "$out/link_${link}_$label.csv"
    vm status
done

echo "== live $(date +%T)"
uv run edge menu live --server "$server" --variant "$DEVICE_VARIANT" \
    --server-model "$CLOUD_MODEL" --plans local,offload,hedge --cancel now \
    --deadline 100 --frames "$frames" --link "$link" --label live \
    --out "$out/menu_live_${link}_live.csv.gz"
vm status

cat >"$out/cloud_session_$link.json" <<EOF
{
  "link": "$link",
  "region": "$region",
  "tries": $try,
  "path": "$path",
  "gpu": "$(grep -m1 -A1 '^name, driver_version' "$work/cloud-$stamp-setup.out" | tail -1)",
  "code": "$(git rev-parse --short HEAD)"
}
EOF
echo "== done $(date +%T)"
