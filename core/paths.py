"""The reference paths.

Deterministic, no RNG -- like everything else in this sandbox, the same choice
always gives the same curve.

A path is a plain dict of dense polyline arrays:

    pts      (N, 2)  world xy of each sample
    tangent  (N, 2)  unit tangent at each sample, pointing along travel
    s        (N,)    cumulative arc length, s[0] = 0
    length   float   total arc length
    name     str     what the UI shows

Samples are uniformly spaced in arc length (~1 cm), which is what lets the
sim's nearest-point search use a fixed index window (sim.py, SEARCH_W).

TWO RULES every path must obey, because the car's spawn pose is fixed:
  1. start at the origin,
  2. with the tangent pointing along +x.
The car spawns at (0, 0, 0.03) with the identity quaternion, so a path obeying
both starts the episode with zero cross-track AND zero heading error. A path
that broke either rule would begin with a step disturbance, which would mix an
initial transient into every tracking metric.

Every curve here is built from three segment primitives -- LINE, ARC and
CORNER -- so the tangents are analytic, including at the joins between
segments, where a numerical gradient would smear across the curvature reversal
and quietly corrupt e_ct at exactly the hardest instant.

  line(length)          straight, curvature 0
  arc(radius, sweep)    constant curvature 1/radius; sweep > 0 turns LEFT
  corner(angle)         a very tight arc (0.15 m) -- see corner() for why it is
                        a fillet rather than a true tangent discontinuity
"""

import numpy as np

SPACING_M = 0.01
# Tightest curve the car is still EXPECTED to manage. `needle` is deliberately
# below this (0.05 m) and is excluded from the turn-authority probe for that
# reason -- it is the control, not a target.
MIN_RADIUS_M = 0.12


def line(length):
    """A straight segment of the given length."""
    return ("line", length)


def arc(radius, sweep_deg):
    """A constant-curvature bend. sweep_deg > 0 turns left."""
    return ("arc", radius, sweep_deg)


CORNER_R = 0.15  # fillet radius used by corner(); see the note below


def corner(angle_deg, radius=CORNER_R):
    """A sharp turn: a very tight arc, not a mathematical corner.

    WHY THIS IS A FILLET AND NOT A ZERO-RADIUS CORNER. A true corner was built
    first and is a dead end, for a reason worth keeping:

      At a tangent discontinuity the car overshoots straight past the apex.
      The nearest path point STAYS at the apex, so the whole offset is
      along-track and `e_ct` -- which is measured perpendicular to the
      reference tangent -- reads EXACTLY ZERO while the car is a metre off the
      path. The steering PID therefore sees no error at all, applies no
      correction, and drives straight off the map at every gain setting. Not a
      tuning failure: a blind spot in the error definition, which a lookahead
      or a pure-pursuit reference would not have.

    So a corner here is a 0.15 m fillet -- half the car's length, five times
    tighter than anything else in the catalogue, and still tight enough that
    tracking it well is genuinely hard -- but with a defined tangent all the
    way round, so the controller gets feedback and gains still matter.
    """
    return ("arc", radius, angle_deg)


def vertex(included_deg, radius=CORNER_R, left=True):
    """A corner named by the ANGLE BETWEEN THE TWO LEGS, not by heading change.

    That is how a corner is normally described on a drawing: two straights
    meeting at `included_deg`. A square corner is 90 deg included; anything
    below that is acute and the path partly doubles back on itself.

        included 90  ->  heading turns  90 deg
        included 60  ->  heading turns 120 deg
        included 30  ->  heading turns 150 deg   (nearly a U-turn)

    Smaller included angle = sharper corner = more heading to unwind in the
    same arc length.
    """
    turn = 180.0 - included_deg
    return arc(radius, turn if left else -turn)


def _line_segment(start_xy, heading, length, spacing=SPACING_M):
    """Straight run from start_xy along heading."""
    n = max(int(round(length / spacing)) + 1, 2)
    step = np.linspace(0.0, length, n)
    direction = np.array((np.cos(heading), np.sin(heading)))
    pts = np.asarray(start_xy, dtype=float) + step[:, None] * direction
    tangent = np.tile(direction, (n, 1))
    return pts, tangent, pts[-1], heading


def _arc_segment(start_xy, start_heading, radius, sweep_deg, spacing=SPACING_M):
    """One circular arc leaving `start_xy` along `start_heading`.

    sweep_deg > 0 turns LEFT (counter-clockwise, +z yaw), < 0 turns right.
    Returns (pts, tangents, end_xy, end_heading). Arc length is radius * sweep,
    so uniform angle spacing IS uniform arc-length spacing.
    """
    sgn = 1.0 if sweep_deg >= 0 else -1.0
    sweep = np.radians(abs(sweep_deg))
    n = max(int(round(radius * sweep / spacing)) + 1, 2)

    # Centre sits one radius to the left of travel for a left turn, right for a
    # right turn. n_left is the heading vector rotated +90 deg.
    n_left = np.array((-np.sin(start_heading), np.cos(start_heading)))
    centre = np.asarray(start_xy, dtype=float) + sgn * radius * n_left

    theta0 = start_heading - sgn * (np.pi / 2.0)  # angle of start, seen from centre
    theta = theta0 + sgn * np.linspace(0.0, sweep, n)

    pts = centre + radius * np.column_stack((np.cos(theta), np.sin(theta)))
    tangent = sgn * np.column_stack((-np.sin(theta), np.cos(theta)))
    return pts, tangent, pts[-1], start_heading + sgn * sweep


def _chain(segments, name, spacing=SPACING_M):
    """Run a list of segment primitives end to end from the origin.

    Each segment starts where the last one ended, heading where it was heading.
    Across line and arc joins the path is C1-continuous -- position and tangent
    both match -- but curvature is not, and an instantaneous curvature reversal
    is a step disturbance to the steering loop. On the slalom and figure-8 that
    step is the most informative moment in the run.

    A corner fillet is just a very tight arc, so this stays true there too --
    the curvature step is simply much larger.
    """
    xy = np.zeros(2)
    heading = 0.0  # rule 2: start tangent to +x
    all_pts, all_tan = [], []
    for seg in segments:
        kind = seg[0]
        if kind == "line":
            pts, tan, xy, heading = _line_segment(xy, heading, seg[1], spacing)
        else:
            pts, tan, xy, heading = _arc_segment(xy, heading, seg[1], seg[2], spacing)
        if all_pts:  # the join point is already the previous segment's last sample
            pts, tan = pts[1:], tan[1:]
        all_pts.append(pts)
        all_tan.append(tan)

    pts = np.vstack(all_pts)
    tangent = np.vstack(all_tan)
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(step)))
    return {"pts": pts, "tangent": tangent, "s": s,
            "length": float(s[-1]), "name": name}


# --- the catalogue ---------------------------------------------------------
#
# Radii are chosen so the tightest curve in every path stays above 1 m (see the
# module docstring), and so the total length is something the car can finish
# inside a sane step budget at ~0.5 m/s.

def gentle_arc():
    """One 3 m radius bend, 120 deg left. The calibration baseline: a single
    constant-curvature demand, nothing transient except the start."""
    return _chain([arc(3.0, 120.0)], "gentle arc — R 3 m, 120° left")


def s_curve():
    """Lane change: 60 deg left then 60 deg right at 3 m radius. Adds ONE
    curvature reversal to the arc -- the first place a lagging controller
    overshoots to the wrong side."""
    return _chain([arc(3.0, 60.0), arc(3.0, -60.0)],
                  "S-curve — R 3 m, 60° left then right")


def u_turn():
    """180 deg left at 2 m radius. Same idea as the arc but a tighter, longer
    sustained demand -- this is where a too-low Kp settles at a steady offset
    and simply never recovers."""
    return _chain([arc(2.0, 180.0)], "U-turn — R 2 m, 180° left")


def slalom():
    """Left 90, right 180, left 90 at 1.5 m radius: three curvature reversals,
    tight enough that the steering channel starts fighting the speed channel
    for wheel authority. Ends heading +x again."""
    return _chain([arc(1.5, 90.0), arc(1.5, -180.0), arc(1.5, 90.0)],
                  "slalom — R 1.5 m, 3 reversals")


def figure_eight():
    """A full left circle then a full right circle, both 1.5 m radius, tangent
    at the origin. The longest smooth path here (~19 m).

    This is the one that justifies the monotonic search window in sim.py: the
    path crosses itself at the origin, so a global nearest-point search would
    snap the reference back to s = 0 every time the car passed through, and
    progress would never advance past the first loop.
    """
    return _chain([arc(1.5, 360.0), arc(1.5, -360.0)],
                  "figure-8 — two R 1.5 m loops")


# --- the hard half: curvature the controller cannot keep up with ------------

def hairpin():
    """Straight, a 0.5 m radius 180, straight. The tightest FINITE curvature
    that is still physically trackable.

    Nothing here is impossible -- the car's own turn radius is ~0.08 m -- but
    the curvature step at each end of the bend is huge, and the controller
    meets it with no warning because there is no lookahead. Slow the car down
    and a well-tuned gain set still tracks it; leave the speed up and it does
    not. That speed dependence is the useful part.
    """
    return _chain([line(1.2), arc(0.5, 180.0), line(1.2)],
                  "hairpin — R 0.5 m, 180°")


def spiral():
    """Seven 90 deg arcs of shrinking radius: 3.0, 2.0, 1.4, 1.0, 0.7, 0.5,
    0.35 m, all turning left.

    The most useful path in the catalogue for the thesis, because it turns
    "the controller cannot follow this" into a NUMBER: the error grows coil by
    coil, so the radius at which a given gain set breaks down can be read
    straight off the cross-track chart. Re-tune and read the new radius. That
    is a limitation curve, not an anecdote.
    """
    radii = (3.0, 2.0, 1.4, 1.0, 0.7, 0.5, 0.35)
    return _chain([arc(r, 90.0) for r in radii],
                  "spiral — R 3.0 → 0.35 m, tightening")


def zigzag():
    """Four right-angle corners (0.15 m fillets) joined by 1.2 m straights.

    The curvature step at each corner is enormous -- 0 to 6.7 1/m and back
    within a few centimetres of travel -- and the controller meets it with no
    lookahead, so an overshoot into the outside of every turn is guaranteed.
    Tuning changes the SIZE of the spike and how fast it recovers; it cannot
    remove it. The straights in between are long enough to let a good gain set
    settle and a bad one still be ringing when the next corner arrives.
    """
    return _chain([line(1.2), corner(90.0), line(1.2), corner(-90.0),
                   line(1.2), corner(90.0), line(1.2), corner(-90.0),
                   line(1.2)],
                  "zigzag — 4 sharp 90° corners")


def _zigzag_of(included_deg, radius, name, leg=1.5, corners=3):
    """Alternating left/right corners of a fixed included angle, joined by
    straights. Alternating keeps it a zigzag rather than a coil, and keeps the
    steering integral from accumulating in one direction."""
    segs = [line(leg)]
    for i in range(corners):
        segs.append(vertex(included_deg, radius, left=(i % 2 == 0)))
        segs.append(line(leg))
    return _chain(segs, name)


def zigzag60():
    """Three 60 deg included corners -- acute, 120 deg of heading to unwind at
    each one, in a 0.12 m arc. Half a metre of travel either side of the apex
    is all the controller gets."""
    return _zigzag_of(60.0, 0.12, "zigzag 60° — acute corners, R 0.12 m")


def zigzag30():
    """Three 30 deg included corners: 150 deg of heading change each, so the
    path very nearly doubles back on itself at every apex.

    Two things to watch here, both real and both documented in the README:
    the steering channel needs full authority through the apex (so the speed
    channel loses it), and the outgoing leg passes within ~0.23 m of the
    incoming one, which is close enough to matter to a nearest-point
    projection with no notion of which leg it is supposed to be on.
    """
    return _zigzag_of(30.0, 0.12, "zigzag 30° — very acute, R 0.12 m")


def needle():
    """30 deg included corners at a 0.05 m radius -- below the turn radius the
    car can achieve at its target speed.

    This is the deliberately impossible one. Tuning cannot fix it because the
    demand is outside the vehicle's envelope, not outside the controller's:
    the only things that help are slowing down (a diff-drive's minimum turn
    radius goes to zero as speed goes to zero) or a bigger actuator. Use it as
    the reference point for what "the plant cannot do this" looks like on the
    charts, so it is not confused with "the gains are wrong".
    """
    return _zigzag_of(30.0, 0.05, "needle — 30° corners, R 0.05 m (impossible)")


def square():
    """A closed 2 m square: four right-angle corners (0.15 m fillets).

    The same impossibility as the zigzag but with every corner turning the
    same way, so the steering integral never gets a sign change to unwind
    against. With any Ki at all this is where windup shows itself.
    """
    return _chain([line(2.0), corner(90.0), line(2.0), corner(90.0),
                   line(2.0), corner(90.0), line(2.0)],
                  "square — 2 m sides, 90° corners")


def straight(length=6.0, spacing=SPACING_M):
    """A straight line along +x. Not in the UI menu -- it exists for the sign
    and convergence probes in experiments/probes.py, where a curving reference
    would confound a pure sign test."""
    n = int(round(length / spacing)) + 1
    s = np.linspace(0.0, length, n)
    return {
        "pts": np.column_stack((s, np.zeros(n))),
        "tangent": np.column_stack((np.ones(n), np.zeros(n))),
        "s": s,
        "length": float(length),
        "name": f"straight {length:g} m",
    }


# Menu order is deliberate: it runs easiest to hardest, which is also the order
# a Kp sweep stops being boring in. The last three are the ones where tuning
# runs out of room -- see HARD.
CATALOGUE = {
    "arc": gentle_arc,
    "scurve": s_curve,
    "uturn": u_turn,
    "slalom": slalom,
    "figure8": figure_eight,
    "hairpin": hairpin,
    "spiral": spiral,
    "zigzag": zigzag,
    "square": square,
    "zigzag60": zigzag60,
    "zigzag30": zigzag30,
    "needle": needle,
}
DEFAULT = "arc"

# Paths whose curvature the controller cannot track exactly at any gain. The
# UI labels these so a large error is read as the path's doing, not the user's.
HARD = {"hairpin", "spiral", "zigzag", "square", "zigzag60", "zigzag30", "needle"}

# Of those, the ones built from corner fillets rather than smooth bends.
SHARP = {"zigzag", "square", "zigzag60", "zigzag30", "needle"}

# Sharper than the car's own turn radius at working speed: the demand is
# outside the VEHICLE's envelope, so no gain set can meet it.
IMPOSSIBLE = {"needle"}


def get(key):
    """Build a path by menu key. Unknown keys fall back to the default rather
    than raising -- a stale value in the browser's UI state must not 500."""
    return CATALOGUE.get(key, CATALOGUE[DEFAULT])()


def options():
    """[(key, display name)] for the UI dropdown, in catalogue order."""
    return [(key, build()["name"]) for key, build in CATALOGUE.items()]


if __name__ == "__main__":  # sanity table
    print(f"{'key':>8} {'length m':>9} {'pts':>6} {'start':>16} "
          f"{'start tangent':>16} {'spacing mm':>10}")
    for key, build in CATALOGUE.items():
        p = build()
        step = np.diff(p["s"])
        print(f"{key:>8} {p['length']:>9.2f} {len(p['pts']):>6} "
              f"({p['pts'][0, 0]:+.3f},{p['pts'][0, 1]:+.3f}) "
              f"({p['tangent'][0, 0]:+.3f},{p['tangent'][0, 1]:+.3f}) "
              f"{1000 * step.min():>5.1f}-{1000 * step.max():.1f}")
