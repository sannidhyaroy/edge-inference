#!/usr/bin/env bash
# Hold chosen cores at a fixed clock for device profiling, and put everything back.
#
#   device_config.sh apply MHZ CPUS [passive]    e.g. apply 1400 2,4 passive   (root)
#   device_config.sh reset                                  (root)
#   device_config.sh show
#
# `apply` records every value it is about to change in a state file, then
# switches boost off for the whole processor and pins each listed core's
# minimum and maximum clock to MHZ under the performance governor. `reset`
# writes the recorded values back and removes the file. A reboot undoes
# everything too, since none of it is persistent.
#
# Order matters when writing limits: a core refuses a minimum above its
# maximum, and the amd-pstate driver refuses an energy preference while the
# performance governor is active.
#
# In its default active mode, amd-pstate lets the processor's firmware choose
# the clock, and it does not go below the base clock (2.0 GHz here) whatever
# the limits say. `passive` hands the clock to Linux first, so a lower limit
# holds. The mode is driver-wide: every core's settings return to their
# defaults when it changes, and `reset` changes it back before restoring the
# recorded values. A reboot restores the default mode too.
set -euo pipefail

STATE=/run/edge-inference-cpu.state
CPU=/sys/devices/system/cpu

need_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "needs root: run through pkexec or sudo" >&2
        exit 1
    fi
}

record() {
    printf '%s %s\n' "$1" "$(cat "$1")" >>"$STATE"
}

STATUS=$CPU/amd_pstate/status

apply() {
    local mhz=$1 cpus=${2//,/ } mode=${3:-} khz=$(($1 * 1000)) dir
    need_root
    if [[ -e $STATE ]]; then
        echo "a configuration is already applied; reset first" >&2
        exit 1
    fi
    : >"$STATE"
    record "$CPU/cpufreq/boost"
    for cpu in $cpus; do
        dir=$CPU/cpu$cpu/cpufreq
        for name in scaling_governor scaling_min_freq scaling_max_freq \
            energy_performance_preference; do
            record "$dir/$name"
        done
    done

    if [[ -n $mode ]]; then
        record "$STATUS"
        echo "$mode" >"$STATUS"
    fi
    echo 0 >"$CPU/cpufreq/boost"
    for cpu in $cpus; do
        dir=$CPU/cpu$cpu/cpufreq
        echo performance >"$dir/scaling_governor"
        if ((khz < $(cat "$dir/scaling_min_freq"))); then
            echo "$khz" >"$dir/scaling_min_freq"
            echo "$khz" >"$dir/scaling_max_freq"
        else
            echo "$khz" >"$dir/scaling_max_freq"
            echo "$khz" >"$dir/scaling_min_freq"
        fi
    done
    echo "applied: boost off; cores $2 held at $mhz MHz${mode:+ (driver mode $mode)}"
}

reset() {
    local path value dir
    need_root
    if [[ ! -e $STATE ]]; then
        echo "nothing applied; nothing to reset"
        return 0
    fi
    declare -A saved
    while read -r path value; do
        saved[$path]=$value
    done <"$STATE"

    if [[ -n ${saved[$STATUS]:-} ]]; then
        echo "${saved[$STATUS]}" >"$STATUS"
    fi
    for path in "${!saved[@]}"; do
        [[ $path == */scaling_governor ]] || continue
        dir=${path%/scaling_governor}
        echo "${saved[$dir/scaling_governor]}" >"$dir/scaling_governor"
        cat "$dir/cpuinfo_max_freq" >"$dir/scaling_max_freq"
        echo "${saved[$dir/scaling_min_freq]}" >"$dir/scaling_min_freq"
        echo "${saved[$dir/scaling_max_freq]}" >"$dir/scaling_max_freq"
        echo "${saved[$dir/energy_performance_preference]}" \
            >"$dir/energy_performance_preference"
    done
    echo "${saved[$CPU/cpufreq/boost]}" >"$CPU/cpufreq/boost"
    rm -f "$STATE"
    echo "reset: every recorded value written back"
}

show() {
    local dir
    echo "boost $(cat "$CPU/cpufreq/boost"); driver mode $(cat "$STATUS" 2>/dev/null || echo -)"
    for dir in "$CPU"/cpu[0-9]*/cpufreq; do
        printf '%-6s %-12s min %7s  max %7s  now %7s kHz  %s\n' \
            "$(basename "$(dirname "$dir")")" "$(cat "$dir/scaling_governor")" \
            "$(cat "$dir/scaling_min_freq")" "$(cat "$dir/scaling_max_freq")" \
            "$(cat "$dir/scaling_cur_freq")" "$(cat "$dir/energy_performance_preference" 2>/dev/null || echo -)"
    done
}

case ${1:-} in
apply) apply "${2:?MHZ}" "${3:?CPUS, e.g. 2,4}" "${4:-}" ;;
reset) reset ;;
show) show ;;
*)
    echo "usage: $0 apply MHZ CPUS [passive] | reset | show" >&2
    exit 2
    ;;
esac
