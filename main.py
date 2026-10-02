# Import modules
import os
import sys

# Import custom modules
import processing
from processing import DataProcessor


# Functions
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


# Main

## Read environment variables and validate inputs

### Mode
mode = os.environ["MODE"]
if mode not in ("curve_fit", "polynomial"):
    fail(f"--mode must be 'curve_fit' or 'polynomial', got {mode!r}")
### Model
model_name = os.environ["MODEL"]
model_function = getattr(processing, model_name, None)
if not callable(model_function):
    fail(f"--model {model_name!r} is not a function defined in processing.py")
### Distances and marks
distances = floats("DISTANCES", "--distances")
marks = floats("MARKS", "--marks")
if not distances or not marks:
    fail("--distances and --marks are required")
if len(distances) != len(marks):
    fail(f"--distances has {len(distances)} values but --marks has {len(marks)}")
### Parameter bounds
lower = floats("LOWER", "--lower")
upper = floats("UPPER", "--upper")
bounds = (lower, upper) if lower and upper else (-float("inf"), float("inf"))
### Correction
correction_mode = "angle"
correction_shots = [None, None, None]
correction = os.environ["CORRECTION"].strip()
if correction:
    parts = [part.strip() for part in correction.split(",")]
    if len(parts) != 4:
        fail(
            "--correction takes MODE,DISTANCE,OLD_MARK,NEW_MARK, " f"got {correction!r}"
        )
    correction_mode = parts[0]
    if correction_mode not in ("angle", "velocity"):
        fail(
            f"--correction mode must be 'angle' or 'velocity', got {correction_mode!r}"
        )
    try:
        correction_shots = [float(part) for part in parts[1:]]
    except ValueError:
        fail(f"--correction distance and marks must be numbers, got {correction!r}")
### Figure size
figsize = floats("FIGSIZE", "--figsize")
if figsize is None or len(figsize) != 2:
    fail("--figsize must be two comma-separated numbers, e.g. 8,6")

## Process data with the validated inputs
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
