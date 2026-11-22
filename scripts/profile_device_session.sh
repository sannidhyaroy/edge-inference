#!/usr/bin/env bash
# Profile the device side at full speed and slowed, unattended, from one prompt.
#
#   pkexec "$PWD/scripts/profile_device_session.sh" [CONFIG ...]
#
# With no arguments every stage runs, one after another: the device
# configurations below, then `edge-twin` (the server models on this processor)
# and `early-exit-sizes` (the early-exit network at smaller inputs).
# Naming some stages runs only those.
#
# Root is used only for what needs it: keeping the machine from sleeping,
# making the energy counters readable, and slowing cores between runs. Every
# profile runs as the user who called pkexec, so the results belong to them.
# On exit, for any reason, the cores are reset and the energy counters get
# back the permissions they had, because readable counters are a known side
# channel. A failed run is logged and the next one still starts.
set -uo pipefail

# name:MHz:cores; an empty MHz leaves the processor as it is, unpinned.
CONFIGS=(
    "laptop-asis-t2::"
    "laptop-2.0g-2c:2000:2,4"
    "laptop-1.4g-2c:1400:2,4"
)
RAPL=(/sys/class/powercap/intel-rapl:0/energy_uj /sys/class/powercap/intel-rapl:0:0/energy_uj)

if [[ $EUID -ne 0 || -z ${PKEXEC_UID:-} ]]; then
    echo "start this through pkexec: pkexec \"\$PWD/scripts/profile_device_session.sh\"" >&2
    exit 1
fi
here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
user=$(id -nu "$PKEXEC_UID")
home=$(getent passwd "$user" | cut -d: -f6)

if [[ -z ${EDGE_SESSION_INHIBITED:-} ]]; then
    # Block suspend, idle sleep and the lid switch until the session ends.
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
        --who=edge-inference --why="device profiling" \
        env EDGE_SESSION_INHIBITED=1 PKEXEC_UID="$PKEXEC_UID" "$0" "$@"
fi

cd "$repo" || exit 1
install -d -o "$user" logs
log=logs/profile-session-$(date +%Y%m%d-%H%M%S).log
install -o "$user" /dev/null "$log"
exec > >(tee -a "$log") 2>&1

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
as_user() { runuser -u "$user" -- env HOME="$home" PATH=/usr/local/bin:/usr/bin:/bin "$@"; }

if pgrep -u "$user" -f "edge menu (answers|profile)" >/dev/null; then
    say "another answers or profile run is busy; not starting"
    exit 1
fi

declare -A modes
for counter in "${RAPL[@]}"; do
    modes[$counter]=$(stat -c %a "$counter")
done
cleanup() {
    "$here/device_config.sh" reset
    for counter in "${!modes[@]}"; do
        chmod "${modes[$counter]}" "$counter"
    done
    say "cleanup: cores reset; energy counters back to ${modes[*]}"
}
trap cleanup EXIT
trap 'say "interrupted"; exit 130' INT TERM

chmod a+r "${RAPL[@]}"
say "energy counters readable for this session"

wanted=("$@")
failed=()
for entry in "${CONFIGS[@]}"; do
    IFS=: read -r name mhz cpus <<<"$entry"
    if ((${#wanted[@]})) && [[ " ${wanted[*]} " != *" $name "* ]]; then
        continue
    fi
    say "== $name"
    "$here/device_config.sh" reset
    pin=()
    if [[ -n $mhz ]]; then
        if ! "$here/device_config.sh" apply "$mhz" "$cpus"; then
            failed+=("$name: apply")
            continue
        fi
        pin=(--cpus "$cpus")
    fi
    "$here/device_config.sh" show

    start=$SECONDS
    as_user uv run --no-sync edge menu profile-device --config "$name" "${pin[@]}" \
        --cool-to 60 --energy || failed+=("$name: profile-device")
    say "profile-device took $(((SECONDS - start) / 60)) min"

    start=$SECONDS
    as_user uv run --no-sync edge menu profile-prepare --config "$name" "${pin[@]}" \
        --cool-to 60 || failed+=("$name: profile-prepare")
    say "profile-prepare took $(((SECONDS - start) / 60)) min"
done

# The server models, timed on this processor at full speed: the edge server is
# a laptop of the same model and processor. The strongest is meant for a GPU;
# its time on a laptop CPU shows why.
if ((${#wanted[@]} == 0)) || [[ " ${wanted[*]} " == *" edge-twin "* ]]; then
    say "== edge-twin: the server models served from this processor"
    "$here/device_config.sh" reset
    for photo in jpeg256q85 jpeg160q85; do
        as_user uv run --no-sync edge menu profile-server --server edge-twin \
            --payloads "data/payloads/$photo" \
            --models resnet50@224/float32,resnet50@224/int8,convnext_t@224/float32,convnext_t@224/int8,convnextv2_b@224/float32 \
            --cool-to 60 || failed+=("edge-twin: $photo")
    done
fi

# The early-exit network at smaller inputs, at full speed and two threads as its
# other profiles were. The trained size is timed again into a file of its own,
# so sizes are compared within one sitting and the drift since is visible.
if ((${#wanted[@]} == 0)) || [[ " ${wanted[*]} " == *" early-exit-sizes "* ]]; then
    say "== early-exit-sizes: the early-exit network at 160, 128, 112 and 96 px"
    "$here/device_config.sh" reset
    for precision in float32 int8; do
        as_user uv run --no-sync edge exits profile --runtime onnxruntime \
            --precision "$precision" --threads 2 --cool-to 60 \
            --out "results/exit_profile_resnet18_early_exit_imagenette_recheck_onnxruntime_${precision}_t2.csv.gz" ||
            failed+=("early-exit-sizes: 160 $precision")
        for size in 128 112 96; do
            as_user uv run --no-sync edge exits profile --runtime onnxruntime \
                --precision "$precision" --threads 2 --image-size "$size" --cool-to 60 ||
                failed+=("early-exit-sizes: $size $precision")
        done
    done
fi

if ((${#failed[@]})); then
    say "finished with failures: ${failed[*]}"
    exit 1
fi
say "finished: every configuration ran"
