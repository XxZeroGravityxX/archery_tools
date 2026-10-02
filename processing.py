# Import modules
import functools
import inspect
import os

import matplotlib.pyplot as plt
import numpy as np

# Import submodules
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import curve_fit

# Functions

## Curve models


### Sine model (simple approximation)
def flexible_sine_model(x, A, B):
    """Return a sine curve with amplitude A and angular frequency B."""
    return A * np.sin(B * x)


### Ballistic model (small-angle scope approximation)
def projectile_range_model(distance, v0, g=9.81, beta=0.0):
    """Return the expected scope mark for a target at ``distance`` meters.

    Derivation (small-angle scope, ballistic range formula on a slope).
    ``distance`` is the slant range measured along the line of sight and
    ``beta`` is the slope angle of that line of sight above the horizontal.
    With ``theta`` the launch angle above the horizontal, the range along the
    slope is:

        R = 2 * v0**2 * cos(theta) * sin(theta - beta) / (g * cos(beta)**2)

    Using ``2 * cos(theta) * sin(theta - beta) = sin(2*theta - beta) - sin(beta)``
    and solving for the angle between the arrow and the line of sight,
    ``angle = theta - beta``:

        angle = 0.5 * (arcsin(R * g * cos(beta)**2 / v0**2 + sin(beta)) - beta)
        mark  = R * angle                             (small-angle scope)

    At ``beta = 0`` this collapses to the flat-ground form
    ``mark = distance * 0.5 * arcsin(distance * g / v0**2)``.

    ``v0`` is the free parameter; ``g`` defaults to 9.81 m/s^2 and ``beta`` to
    a level shot. Both are meant to be fixed before fitting (see
    ``bind_slope_angle``) rather than fitted. ``arcsin`` requires
    ``distance * g * cos(beta)**2 / v0**2 <= 1 - sin(beta)``.

    Parameters
    ----------
    distance : float or array
        Slant range to the target in meters, along the line of sight.
    v0 : float
        Effective initial arrow speed in m/s (fitted).
    g : float
        Gravitational acceleration, defaults to 9.81 m/s^2.
    beta : float
        Slope angle of the line of sight in degrees, positive uphill and
        negative downhill. Defaults to 0 (level shot).
    """
    beta_rad = np.radians(beta)
    angle_rad = 0.5 * (
        np.arcsin(distance * g * np.cos(beta_rad) ** 2 / v0**2 + np.sin(beta_rad))
        - beta_rad
    )
    return distance * angle_rad


### Extended ballistic model for compound bows (small-angle scope approximation)
def compound_bow_mark_model(
    distance,
    v0,
    eye_offset,
    sight_baseline,
    g=9.81,
    beta=0.0,
):
    """Return the expected scope mark for a compound bow at ``distance`` meters.

    Extends the plain ballistic ``projectile_range_model`` with the sight
    geometry of a bow-mounted sight, where the archer's eye is offset
    (vertically and forward) from the arrow launch line. Under small-angle
    approximations (small elevation angle, sight bar much shorter than target
    distance), the mark on the sight tape decomposes into three additive
    contributions:

        mark(d) ≈  g * d * cos(beta) / (2 * v0**2)
                                      (ballistic drop term, grows with d)
                  + eye_offset / d    (sight-geometry term, dominates at short d)
                  + sight_baseline    (fixed pin-to-arrow offset)

    The ``cos(beta)`` factor is the small-angle limit of the slope-corrected
    launch angle in ``projectile_range_model``, i.e. the classic rule that an
    inclined shot needs the mark of its *horizontal* distance. The sight
    geometry term is measured along the line of sight, so the slope leaves it
    unchanged.

    Parameters
    ----------
    distance : float or array
        Slant range to the target in meters, along the line of sight.
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
    beta : float
        Slope angle of the line of sight in degrees, positive uphill and
        negative downhill. Defaults to 0 (level shot).
    """
    ballistic_term = g * distance * np.cos(np.radians(beta)) / (2 * v0**2)
    geometric_term = eye_offset / distance
    return ballistic_term + geometric_term + sight_baseline


## Slope angle utilities
def accepts_slope_angle(model_function):
    """Return whether ``model_function`` takes a ``beta`` slope angle."""
    try:
        return "beta" in inspect.signature(model_function).parameters
    except (TypeError, ValueError):
        return False


def bind_slope_angle(model_function, beta=0.0):
    """Return ``model_function`` with its ``beta`` slope angle fixed.

    Binding the slope after fitting keeps it separate from fitted parameters
    such as velocity. A level shot (``beta`` of 0) already matches the model
    defaults, so the function is returned untouched.
    """
    if model_function is None or not beta:
        return model_function

    if not accepts_slope_angle(model_function):
        name = getattr(model_function, "__name__", repr(model_function))
        raise ValueError(f"Model {name} does not accept a 'beta' slope angle.")

    return functools.partial(model_function, beta=beta)


## Mark corrections
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


# Classes


## Curve fitting
class CurveFitModel:
    """Store a curve function and its optimized SciPy parameters."""

    def __init__(self, model_function, parameters, covariance):
        self.model_function = model_function
        self.parameters = parameters
        self.covariance = covariance

    def __call__(self, x):
        return self.model_function(x, *self.parameters)


## Data processing
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
        beta=0.0,
    ):
        """Fit and store a polynomial or SciPy curve model.

        ``beta`` is the slope angle of the line of sight in degrees (positive
        uphill, negative downhill). Curve parameters are fitted against the
        level-shot model, then ``beta`` is bound to the stored model for
        evaluation. This prevents the optimizer from absorbing the slope into
        velocity and only applies to ``mode='curve_fit'``.
        """
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
            evaluated_model_function = bind_slope_angle(model_function, beta)
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
                model_function=evaluated_model_function,
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

    @staticmethod
    def _title_extras(textra):
        """Return ``textra`` rendered as a ``" (name=value, ...)"`` title suffix."""
        if not textra:
            return ""
        pairs = ", ".join(
            f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}"
            for name, value in textra.items()
        )
        return f" ({pairs})"

    def plot_data(
        self,
        data_distances=None,
        data_marks=None,
        plot_distances=None,
        plot_marks=None,
        textra=None,
        output_path=".",
        output_fname="fit_model_plot",
        output_fext="png",
        figsize=(8, 6),
    ):
        """Plot explicit data or data produced by previous method calls.

        ``textra`` is a mapping of extra values to annotate the title with, so
        ``{"beta": 3}`` renders as ``... — model_name (beta=3)``. The image is
        written to ``output_path/output_fname.<model name>.output_fext``.
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
        ax.set_title(f"Fitted Model Plot — {model_name}{self._title_extras(textra)}")
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
        beta=0.0,
        correction_mode="angle",
        correction_distance=None,
        correction_old_mark=None,
        correction_new_mark=None,
        poly_model=None,
        start=0,
        end=50,
        step=1,
        show_plot=True,
        textra=None,
        output_path=".",
        output_fname="fit_model_plot",
        output_fext="png",
        figsize=(8, 6),
    ):
        """Run the fit, correction, evaluation, and optional plotting pipeline.

        The correction is derived here via ``compute_correction`` from the
        ``correction_*`` arguments. It is skipped unless the distance and both
        marks are given. ``beta`` is the slope angle of the line of sight in
        degrees applied after fitting the level-shot curve; slope-aware curve
        models also get it added to the ``textra`` plot title annotations.
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
                beta=beta,
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
            title_extras = dict(textra) if textra else {}
            if mode == "curve_fit" and accepts_slope_angle(model_function):
                title_extras.setdefault("beta", beta)
            self.plot_data(
                data_distances=self.data_distances,
                data_marks=self.data_marks,
                plot_distances=self.plot_distances,
                plot_marks=self.plot_marks,
                textra=title_extras,
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
