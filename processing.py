# Import modules
import os

import matplotlib.pyplot as plt
import numpy as np

# Import submodules
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit


# Curve fit model
class CurveFitModel:
    """Store a curve function and its optimized SciPy parameters."""

    def __init__(self, model_function, parameters, covariance):
        self.model_function = model_function
        self.parameters = parameters
        self.covariance = covariance

    def __call__(self, x):
        return self.model_function(x, *self.parameters)


# Sample curve models
def flexible_sine_model(x, A, B):
    """Return a sine curve with amplitude A and angular frequency B."""
    return A * np.sin(B * x)


def projectile_range_model(distance, v0, g=9.81):
    """Return the expected scope mark for a target at ``distance`` meters.

    Derivation (small-angle scope, ballistic range formula):

        mark  ~=  distance * angle                   (small-angle scope)
        R      =  (v0**2 / g) * sin(2 * angle)       (ballistic range)

    Solving the second equation for the launch angle and substituting:

        angle  =  0.5 * arcsin(distance * g / v0**2)
        mark   =  distance * 0.5 * arcsin(distance * g / v0**2)

    ``v0`` is the free parameter; ``g`` defaults to 9.81 m/s^2 but can be
    overridden (e.g. fixed via ``functools.partial`` before passing to
    ``curve_fit``, or fitted jointly if left in the signature). ``arcsin``
    requires ``distance * g / v0**2 <= 1``, i.e. ``v0 >= sqrt(distance * g)``.
    """
    angle_rad = 0.5 * np.arcsin(distance * g / v0**2)
    return distance * angle_rad


def compound_bow_mark_model(
    distance,
    v0,
    eye_offset,
    sight_baseline,
    g=9.81,
):
    """Return the expected scope mark for a compound bow at ``distance`` meters.

    Extends the plain ballistic ``projectile_range_model`` with the sight
    geometry of a bow-mounted sight, where the archer's eye is offset
    (vertically and forward) from the arrow launch line. Under small-angle
    approximations (level target, small elevation angle, sight bar much
    shorter than target distance), the mark on the sight tape decomposes
    into three additive contributions:

        mark(d) ≈  g * d / (2 * v0**2)      (ballistic drop term, grows with d)
                  + eye_offset / d          (sight-geometry term, dominates at short d)
                  + sight_baseline          (fixed pin-to-arrow offset)

    Parameters
    ----------
    distance : float or array
        Horizontal distance to the target in meters (same elevation).
    v0 : float
        Effective initial arrow speed in m/s (fitted).
    eye_offset : float
        Lumped eye/pin geometric offset in meters (fitted). Captures the
        product of the archer's eye-above-arrow height and the sight-bar
        length; its ``1 / distance`` contribution dominates at close range.
    sight_baseline : float
        Constant sight-tape offset in the same units as ``mark`` (fitted).
    g : float
        Gravitational acceleration, defaults to 9.81 m/s^2.
    """
    ballistic_term = g * distance / (2 * v0**2)
    geometric_term = eye_offset / distance
    return ballistic_term + geometric_term + sight_baseline


# Mark corrections
def compute_correction(
    distance,
    old_mark,
    new_mark,
    mode="angle",
):
    """Return the correction value that maps old sight marks onto new ones.

    Both marks must come from the same ``distance``, otherwise the
    distance-dependent ballistic part of the angle would leak into the
    correction. They are normalized into angle space (``mark / distance``) and
    then compared according to ``mode``:

    - ``"angle"``: peep/sight movement is a constant angular offset, so the
      correction is the difference ``new_angle - old_angle``.
    - ``"velocity"``: a speed change scales the angle by ``(v0 / v1) ** 2``
      under the small-angle approximation, so the correction is the quotient
      ``new_angle / old_angle``.
    """
    distance = np.asarray(distance, dtype=float)
    old_angle = np.asarray(old_mark, dtype=float) / distance
    new_angle = np.asarray(new_mark, dtype=float) / distance

    if mode == "angle":
        return new_angle - old_angle
    if mode == "velocity":
        return new_angle / old_angle
    raise ValueError("mode must be either 'angle' or 'velocity'.")


def apply_correction(distances, marks, correction):
    """Apply angular and/or velocity corrections to evaluated marks.

    ``correction`` accepts ``{"angle": delta_theta}``, ``{"velocity": factor}``,
    or both. The velocity factor is multiplicative and is applied first; the
    angular offset is additive in angle space, so its effect on the mark scales
    linearly with distance (``mark += distance * delta_theta``).
    """
    if not correction:
        return marks

    unknown_keys = set(correction) - {"angle", "velocity"}
    if unknown_keys:
        raise ValueError(
            f"Unknown correction keys: {sorted(unknown_keys)}. "
            "Expected 'angle' and/or 'velocity'."
        )

    corrected_marks = np.asarray(marks, dtype=float)
    if "velocity" in correction:
        corrected_marks = corrected_marks * correction["velocity"]
    if "angle" in correction:
        corrected_marks = corrected_marks + (
            np.asarray(distances, dtype=float) * correction["angle"]
        )

    return corrected_marks


# Data processor
class DataProcessor:
    """Fit, evaluate, and plot polynomial or curve-fitted data."""

    def __init__(self, poly_model=None):
        self.poly_model = poly_model
        self.data_distances = None
        self.data_marks = None
        self.plot_distances = None
        self.plot_marks = None

    def fit_data(
        self,
        data_distances=(10, 18, 20, 25, 30, 50),
        data_marks=(3, 3.4, 3.5, 3.9, 4.2, 5.9),
        mode="polynomial",
        degree=2,
        model_function=None,
        initial_parameters=None,
        bounds=(-np.inf, np.inf),
        maxfev=None,
    ):
        """Fit and store a polynomial or SciPy curve model."""
        self.data_distances = np.asarray(data_distances)
        self.data_marks = np.asarray(data_marks)

        if mode == "polynomial":
            self.poly_model = Polynomial.fit(
                self.data_distances,
                self.data_marks,
                deg=degree,
            )
        elif mode == "curve_fit":
            if model_function is None:
                model_function = flexible_sine_model
            curve_fit_arguments = {
                "f": model_function,
                "xdata": self.data_distances,
                "ydata": self.data_marks,
                "p0": initial_parameters,
                "bounds": bounds,
            }
            if maxfev is not None:
                curve_fit_arguments["maxfev"] = maxfev

            parameters, covariance = curve_fit(**curve_fit_arguments)
            self.poly_model = CurveFitModel(
                model_function=model_function,
                parameters=parameters,
                covariance=covariance,
            )
            print(f"Optimized Parameters: {parameters}")
        else:
            raise ValueError("mode must be either 'polynomial' or 'curve_fit'.")

        print("Sample Data:")
        for distance, mark in zip(self.data_distances, self.data_marks):
            print(f"Distance: {distance}, Mark: {mark}")

        return self.data_distances, self.data_marks, self.poly_model

    def eval_data(self, poly_model=None, start=0, end=50, step=1, correction=None):
        """Evaluate an explicit or previously stored polynomial model."""
        model = poly_model if poly_model is not None else self.poly_model
        if model is None:
            raise ValueError("No polynomial model found. Call fit_data first.")

        self.poly_model = model
        self.plot_distances = np.arange(start, end + step, step)
        self.plot_marks = apply_correction(
            self.plot_distances,
            model(self.plot_distances),
            correction,
        )

        print("Evaluated Data:")
        for distance, mark in zip(self.plot_distances, self.plot_marks):
            print(f"Distance: {distance}, Mark: {round(mark, 2)}")

        return self.plot_distances, self.plot_marks

    def _model_name(self):
        """Return a short identifier for the currently stored model."""
        model = self.poly_model
        if model is None:
            return "unknown"
        fn = getattr(model, "model_function", None)
        if fn is not None:
            name = getattr(fn, "__name__", None)
            if name is None:
                name = getattr(getattr(fn, "func", None), "__name__", "curve_fit")
            return name
        if isinstance(model, Polynomial):
            return f"polynomial_deg{model.degree()}"
        return type(model).__name__

    def plot_data(
        self,
        data_distances=None,
        data_marks=None,
        plot_distances=None,
        plot_marks=None,
        output_path=".",
        output_fname="fit_model_plot",
        output_fext="png",
        figsize=(8, 6),
    ):
        """Plot explicit data or data produced by previous method calls.

        The image is written to
        ``output_path/output_fname.<model name>.output_fext``.
        """
        data_distances = (
            self.data_distances if data_distances is None else data_distances
        )
        data_marks = self.data_marks if data_marks is None else data_marks
        plot_distances = (
            self.plot_distances if plot_distances is None else plot_distances
        )
        plot_marks = self.plot_marks if plot_marks is None else plot_marks

        if any(
            value is None
            for value in (data_distances, data_marks, plot_distances, plot_marks)
        ):
            raise ValueError("No plot data found. Call fit_data and eval_data first.")

        model_name = self._model_name()
        output_file = os.path.join(
            output_path,
            f"{output_fname}.{model_name}.{output_fext.lstrip('.')}",
        )
        if output_path:
            os.makedirs(output_path, exist_ok=True)

        plt.close("all")
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(data_distances, data_marks, color="red", label="Original Data")
        ax.plot(
            plot_distances,
            plot_marks,
            color="blue",
            label=f"Fitted Model",
        )
        ax.set_title(f"Fitted Model Plot — {model_name}")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Mark")
        ax.legend()
        ax.grid(True)
        fig.savefig(output_file)
        print(f"Saved plot: {output_file}")
        plt.show()

        return fig, ax

    def process_data(
        self,
        data_distances=(10, 18, 20, 25, 30, 50),
        data_marks=(3, 3.4, 3.5, 3.9, 4.2, 5.9),
        mode="polynomial",
        degree=2,
        model_function=None,
        initial_parameters=None,
        bounds=(-np.inf, np.inf),
        maxfev=None,
        correction_mode="angle",
        correction_distance=None,
        correction_old_mark=None,
        correction_new_mark=None,
        poly_model=None,
        start=0,
        end=50,
        step=1,
        show_plot=True,
        output_path=".",
        output_fname="fit_model_plot",
        output_fext="png",
        figsize=(8, 6),
    ):
        """Run the fit, correction, evaluation, and optional plotting pipeline.

        The correction is derived here via ``compute_correction`` from the
        ``correction_*`` arguments. It is skipped unless the distance and both
        marks are given.
        """
        correction = None
        correction_arguments = (
            correction_distance,
            correction_old_mark,
            correction_new_mark,
        )
        if any(value is not None for value in correction_arguments):
            if None in correction_arguments:
                raise ValueError(
                    "A correction needs correction_distance, "
                    "correction_old_mark and correction_new_mark."
                )
            correction_value = compute_correction(
                distance=correction_distance,
                old_mark=correction_old_mark,
                new_mark=correction_new_mark,
                mode=correction_mode,
            )
            correction = {correction_mode: correction_value}
            print(f"Computed Correction ({correction_mode}): {correction_value}")

        if poly_model is not None:
            self.poly_model = poly_model

        if self.poly_model is None:
            self.fit_data(
                data_distances=data_distances,
                data_marks=data_marks,
                mode=mode,
                degree=degree,
                model_function=model_function,
                initial_parameters=initial_parameters,
                bounds=bounds,
                maxfev=maxfev,
            )
        else:
            self.data_distances = np.asarray(data_distances)
            self.data_marks = np.asarray(data_marks)

        self.eval_data(
            poly_model=self.poly_model,
            start=start,
            end=end,
            correction=correction,
            step=step,
        )

        if show_plot:
            self.plot_data(
                data_distances=self.data_distances,
                data_marks=self.data_marks,
                plot_distances=self.plot_distances,
                plot_marks=self.plot_marks,
                output_path=output_path,
                output_fname=output_fname,
                output_fext=output_fext,
                figsize=figsize,
            )

        return (
            self.data_distances,
            self.data_marks,
            self.poly_model,
            self.plot_distances,
            self.plot_marks,
        )
