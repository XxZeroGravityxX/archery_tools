#!/usr/bin/env bash
#
# Thin CLI wrapper around DataProcessor.process_data in processing.py.
set -euo pipefail

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# --- Defaults ---
DISTANCES="10,20,30"
MARKS="2.5,2.7,3.4"
MODE="curve_fit"
DEGREE="2"
MODEL="compound_bow_mark_model"
P0="10.0,0.02,2.0"
LOWER="0,0,-inf"
UPPER="inf,inf,inf"
MAXFEV=""
BETA="0"
CORRECTION=""
START="1"
END="50"
STEP="1"
SHOW_PLOT="1"
OUTPUT_PATH="."
OUTPUT_FNAME="fit_model_plot"
OUTPUT_FEXT="png"
FIGSIZE="8,6"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<EOF
${SCRIPT_NAME} - fit, evaluate and plot archery sight marks.

Usage:
  ${SCRIPT_NAME} [options]

Data options:
  -d, --distances LIST Comma-separated target distances in meters.
                       (default: ${DISTANCES})
  -m, --marks LIST     Comma-separated measured sight marks, one per distance.
                       (default: ${MARKS})
  --mode MODE          Fitting strategy: 'curve_fit' or 'polynomial'.
                       (default: ${MODE})
  --degree N           Polynomial degree. Only used when --mode polynomial.
                       (default: ${DEGREE})
  --model NAME         Name of a model function defined in processing.py, e.g.
                       compound_bow_mark_model, projectile_range_model,
                       flexible_sine_model. Only used when --mode curve_fit.
                       (default: ${MODEL})
  --p0 LIST            Comma-separated initial parameter guesses for the model.
                       For compound_bow_mark_model: v0 [m/s], eye_offset [m],
                       sight_baseline [mark units]. (default: ${P0})
  --lower LIST         Comma-separated lower bounds for --p0, one per parameter.
                       'inf' and '-inf' accepted. (default: ${LOWER})
  --upper LIST         Comma-separated upper bounds for --p0, one per parameter.
                       (default: ${UPPER})
  --maxfev N           Max curve_fit function evaluations. Unset means SciPy's
                       own default.
  -b, --beta DEG       Slope angle of the line of sight in degrees, positive
                       uphill and negative downhill. Distances are then read as
                       slant ranges along that line of sight. Applied to the
                       fitted level-shot curve in --mode curve_fit.
                       (default: ${BETA})

Correction options:
  -c, --correction LIST
                       A reference shot taken before and after a bow change,
                       as MODE,DISTANCE,OLD_MARK,NEW_MARK. Both marks must come
                       from the same DISTANCE. The correction is computed from
                       them and applied to the evaluated marks. MODE is 'angle'
                       for a peep or sight move (additive offset) or 'velocity'
                       for an arrow speed change (multiplicative factor).
                       Omit the flag to skip the correction.
                       Example: -c angle,30,3.4,3.2

Evaluation options:
  --start N            First distance to evaluate, in meters. Keep it above 0:
                       compound_bow_mark_model divides by distance and returns
                       inf at 0. (default: ${START})
  --end N              Last distance to evaluate, in meters. (default: ${END})
  --step N             Distance increment in meters. (default: ${STEP})

Output options:
  --output-path DIR    Folder to write the plot image into; created if missing.
                       (default: ${OUTPUT_PATH})
  --output-fname NAME  Base name of the image. The model name is appended, so
                       the final file is NAME.MODEL.EXT, e.g.
                       ${OUTPUT_FNAME}.${MODEL}.${OUTPUT_FEXT}
                       (default: ${OUTPUT_FNAME})
  --output-fext EXT    Image extension, e.g. png, svg, pdf.
                       (default: ${OUTPUT_FEXT})
  --figsize W,H        Figure size in inches. (default: ${FIGSIZE})
  --no-plot            Skip plotting; print the fitted and evaluated data only.
  --python BIN         Python interpreter to use. (default: ${PYTHON_BIN})
  -h, --help           Show this help and exit.

Examples:
  ${SCRIPT_NAME}
  ${SCRIPT_NAME} -d 18,30,50 -m 2.6,3.4,5.1 --end 60
  ${SCRIPT_NAME} --mode polynomial --degree 2 --no-plot
  ${SCRIPT_NAME} -b 15
  ${SCRIPT_NAME} -c angle,30,3.4,3.2
  ${SCRIPT_NAME} --output-path plots --output-fname session01 --output-fext svg
EOF
}

require_value() {
    if [[ -z "${2:-}" ]]; then
        echo "${SCRIPT_NAME}: option $1 requires a value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    # Normalize --flag=value into --flag value.
    if [[ "$1" == --*=* ]]; then
        set -- "${1%%=*}" "${1#*=}" "${@:2}"
    fi

    case "$1" in
        -d|--distances) require_value "$1" "${2:-}"; DISTANCES="$2"; shift 2 ;;
        -m|--marks) require_value "$1" "${2:-}"; MARKS="$2"; shift 2 ;;
        --mode) require_value "$1" "${2:-}"; MODE="$2"; shift 2 ;;
        --degree) require_value "$1" "${2:-}"; DEGREE="$2"; shift 2 ;;
        --model) require_value "$1" "${2:-}"; MODEL="$2"; shift 2 ;;
        --p0) require_value "$1" "${2:-}"; P0="$2"; shift 2 ;;
        --lower) require_value "$1" "${2:-}"; LOWER="$2"; shift 2 ;;
        --upper) require_value "$1" "${2:-}"; UPPER="$2"; shift 2 ;;
        --maxfev) require_value "$1" "${2:-}"; MAXFEV="$2"; shift 2 ;;
        -b|--beta) require_value "$1" "${2:-}"; BETA="$2"; shift 2 ;;
        -c|--correction) require_value "$1" "${2:-}"; CORRECTION="$2"; shift 2 ;;
        --start) require_value "$1" "${2:-}"; START="$2"; shift 2 ;;
        --end) require_value "$1" "${2:-}"; END="$2"; shift 2 ;;
        --step) require_value "$1" "${2:-}"; STEP="$2"; shift 2 ;;
        --output-path) require_value "$1" "${2:-}"; OUTPUT_PATH="$2"; shift 2 ;;
        --output-fname) require_value "$1" "${2:-}"; OUTPUT_FNAME="$2"; shift 2 ;;
        --output-fext) require_value "$1" "${2:-}"; OUTPUT_FEXT="$2"; shift 2 ;;
        --figsize) require_value "$1" "${2:-}"; FIGSIZE="$2"; shift 2 ;;
        --no-plot) SHOW_PLOT="0"; shift ;;
        --python) require_value "$1" "${2:-}"; PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "${SCRIPT_NAME}: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"

export DISTANCES MARKS MODE DEGREE MODEL P0 LOWER UPPER MAXFEV \
    BETA CORRECTION START END STEP SHOW_PLOT \
    OUTPUT_PATH OUTPUT_FNAME OUTPUT_FEXT FIGSIZE

"$PYTHON_BIN" - <<'PY'
import os
import sys

import processing
from processing import DataProcessor


def fail(message):
    sys.exit(f"fit_marks: {message}")


def floats(name, label):
    raw = os.environ[name].strip()
    if not raw:
        return None
    try:
        return tuple(float(part) for part in raw.split(","))
    except ValueError:
        fail(f"{label} must be a comma-separated list of numbers, got {raw!r}")


def number(name, label, cast=float):
    raw = os.environ[name].strip()
    try:
        return cast(raw)
    except ValueError:
        fail(f"{label} must be a number, got {raw!r}")


mode = os.environ["MODE"]
if mode not in ("curve_fit", "polynomial"):
    fail(f"--mode must be 'curve_fit' or 'polynomial', got {mode!r}")

model_name = os.environ["MODEL"]
model_function = getattr(processing, model_name, None)
if not callable(model_function):
    fail(f"--model {model_name!r} is not a function defined in processing.py")

distances = floats("DISTANCES", "--distances")
marks = floats("MARKS", "--marks")
if not distances or not marks:
    fail("--distances and --marks are required")
if len(distances) != len(marks):
    fail(f"--distances has {len(distances)} values but --marks has {len(marks)}")

lower = floats("LOWER", "--lower")
upper = floats("UPPER", "--upper")
bounds = (lower, upper) if lower and upper else (-float("inf"), float("inf"))

correction_mode = "angle"
correction_shots = [None, None, None]
correction = os.environ["CORRECTION"].strip()
if correction:
    parts = [part.strip() for part in correction.split(",")]
    if len(parts) != 4:
        fail(
            "--correction takes MODE,DISTANCE,OLD_MARK,NEW_MARK, "
            f"got {correction!r}"
        )
    correction_mode = parts[0]
    if correction_mode not in ("angle", "velocity"):
        fail(f"--correction mode must be 'angle' or 'velocity', got {correction_mode!r}")
    try:
        correction_shots = [float(part) for part in parts[1:]]
    except ValueError:
        fail(f"--correction distance and marks must be numbers, got {correction!r}")

figsize = floats("FIGSIZE", "--figsize")
if figsize is None or len(figsize) != 2:
    fail("--figsize must be two comma-separated numbers, e.g. 8,6")

processor = DataProcessor()
processor.process_data(
    data_distances=distances,
    data_marks=marks,
    mode=mode,
    degree=number("DEGREE", "--degree", int),
    model_function=model_function,
    initial_parameters=floats("P0", "--p0"),
    bounds=bounds,
    maxfev=number("MAXFEV", "--maxfev", int) if os.environ["MAXFEV"].strip() else None,
    beta=number("BETA", "--beta"),
    correction_mode=correction_mode,
    correction_distance=correction_shots[0],
    correction_old_mark=correction_shots[1],
    correction_new_mark=correction_shots[2],
    start=number("START", "--start"),
    end=number("END", "--end"),
    step=number("STEP", "--step"),
    show_plot=os.environ["SHOW_PLOT"] == "1",
    output_path=os.environ["OUTPUT_PATH"],
    output_fname=os.environ["OUTPUT_FNAME"],
    output_fext=os.environ["OUTPUT_FEXT"],
    figsize=figsize,
)
PY
