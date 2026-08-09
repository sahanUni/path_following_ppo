"""Plain PID controller plus an explicit anti-windup variant.

PROVENANCE: verbatim copy of D:\\Msc\\Experiments\\Q1Q2\\pid.py, taken 2026-07-28.
Do not "improve" it. The sandbox exists to show what the thesis controller
actually does, quirks included:
  - no anti-windup: _integral accumulates even while the output is clamped
  - derivative kick: reset() zeroes _prev_error, so the first update after a
    reset sees derivative = error / dt (a huge one-substep spike)
If Q1Q2/pid.py changes, re-copy this file.

PIDController below remains the verbatim thesis implementation.
ConditionalPIDController adds conditional integration for the sandbox and
future RL controllers without erasing that reproducibility baseline.
"""


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self._prev_error = 0.0
        self._integral = 0.0

    def reset(self):
        self._prev_error = 0.0
        self._integral = 0.0

    def update(self, measurement, dt):
        """One PID step: returns the control output for the current measurement."""
        error = self.setpoint - measurement

        p_term = self.kp * error

        self._integral += error * dt
        i_term = self.ki * self._integral

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        output = p_term + i_term + d_term

        min_limit, max_limit = self.output_limits
        if min_limit is not None:
            output = max(min_limit, output)
        if max_limit is not None:
            output = min(max_limit, output)

        self._prev_error = error
        return output


class ConditionalPIDController(PIDController):
    """PID with conditional integration and downstream actuator feedback.

    The base class is intentionally kept verbatim. This subclass prevents a
    new integral increment when either:

    * this PID's own output clamp rejects the requested increment, or
    * apply_output_feedback() reports that a downstream allocator could not
      apply the PID output and the increment pushes further into the limit.

    The second case matters for the path-following mixer: each PID can be
    inside its own [-1, 1] limit while their combined wheel request is not.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_error = 0.0
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.last_unclamped_output = 0.0
        self.last_output = 0.0
        self.last_internal_hold = False
        self.last_downstream_hold = False
        self._last_integral_delta = 0.0

    def reset(self):
        super().reset()
        self.last_error = 0.0
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.last_unclamped_output = 0.0
        self.last_output = 0.0
        self.last_internal_hold = False
        self.last_downstream_hold = False
        self._last_integral_delta = 0.0

    def _clip(self, output):
        min_limit, max_limit = self.output_limits
        if min_limit is not None:
            output = max(min_limit, output)
        if max_limit is not None:
            output = min(max_limit, output)
        return output

    @staticmethod
    def _pushes_further_into_limit(error, requested, applied, tolerance=1e-12):
        rejected = requested - applied
        return abs(rejected) > tolerance and error * rejected > 0.0

    def update(self, measurement, dt):
        """Return one PID output while conditionally accepting its I increment."""
        error = self.setpoint - measurement
        p_term = self.kp * error
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        integral_before = self._integral
        integral_delta = error * dt if self.ki != 0.0 else 0.0
        self._integral += integral_delta
        i_term = self.ki * self._integral
        unclamped = p_term + i_term + d_term
        output = self._clip(unclamped)

        self.last_internal_hold = (
            integral_delta != 0.0
            and self._pushes_further_into_limit(error, unclamped, output)
        )
        if self.last_internal_hold:
            self._integral = integral_before
            integral_delta = 0.0
            i_term = self.ki * self._integral
            unclamped = p_term + i_term + d_term
            output = self._clip(unclamped)

        self.last_error = error
        self.last_p_term = p_term
        self.last_i_term = i_term
        self.last_d_term = d_term
        self.last_unclamped_output = unclamped
        self.last_output = output
        self.last_downstream_hold = False
        self._last_integral_delta = integral_delta
        self._prev_error = error
        return output

    def apply_output_feedback(self, applied_output):
        """Undo this step's I increment if a downstream limit rejected it.

        Call this after control allocation with the portion of the actuator
        command actually assigned to this PID channel. Returns True when the
        increment was rolled back.
        """
        self.last_downstream_hold = (
            self._last_integral_delta != 0.0
            and self._pushes_further_into_limit(
                self.last_error, self.last_output, applied_output
            )
        )
        if self.last_downstream_hold:
            self._integral -= self._last_integral_delta
            self._last_integral_delta = 0.0
        return self.last_downstream_hold
