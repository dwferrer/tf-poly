import tensorflow as tf
import numpy as np


def signed_point_line_area(lines, points):
    """
    Computes the signed (oriented) area of the parallelogram specified by the given directed lines (specified as two
     non-identical points) and each of the given points. A negative value indicates the point is to the right of the line
     while a positive value indicates the point is to the left. The leading shapes of lines and points must be
     broadcastable.
    Args:
        lines: A [..., 2, 2] floating point tensor of lines from point a (coordinates [..., 0, :]) to point b
         (coordinates [..., 1, :]).
        points: A [..., 2] A tensor of 2d point coordintes

    Returns:
        A [..., ] floating point tensor containing the signed area. Its leading shape will be the broadcast shapes
         of lines and points
    """

    x = tf.convert_to_tensor(lines)
    p = tf.convert_to_tensor(points)

    a = ((x[..., 0, 0] - p[..., 0]) * (x[..., 1, 1] - x[..., 0, 1]) -
         (x[..., 1, 0] - x[..., 0, 0]) * (x[..., 0, 1] - p[..., 1]))

    return a


def signed_point_line_distance(lines, points):
    """
    Computes the signed perpendicular 2d point-line distance from each of the given directed lines (specified as two
     non-identical points) to each of the given points. A negative value indicates the point is to the right of the line
     while a positive value indicates the point is to the left. The leading shapes of lines and points must be
     broadcastable.
    Args:
        lines: A [..., 2, 2] floating point tensor of lines from point a (coordinates [..., 0, :]) to point b
         (coordinates [..., 1, :]).
        points: A [..., 2] A tensor of 2d point coordintes

    Returns:
        A [..., ] floating point tensor containing the signed distance. Its leading shape will be the broadcast shapes
         of lines and points
    """
    lines = tf.convert_to_tensor(lines)
    a = signed_point_line_area(lines, points)

    line_length = tf.math.reduce_euclidean_norm(lines[..., 1, :] - lines[..., 0, :], axis=-1)

    return a / line_length


def point_line_segment_distance(lines, points):
    pass


def get_edges(polygons):
    """
    Given 1 or more polygons given as an oriented sequence of 2d points without duplicates, return a tensor containing
     the directed edges.
    Args:
        polygons: A [..., N, 2] tensor containing an oriented sequence of points without duplicate

    Returns:
        A [..., N, 2, 2] tensor containing the directed edges

    """

    a = polygons
    b = tf.roll(polygons, shift=-1, axis=-2)

    return tf.stack([a, b], axis=-2)


def line_segment_intersection(a, b, include_boundary=True):
    """
    Compute the intersections of two sets of line-segments a [..., 2, 2] and b [..., 2, 2], yielding both a boolean
     indicator [...,] of whether an intersection exists and a list of points [..., 2] which are the
     intersections for points where the indicator is true, otherwise arbitrarily valued. The leading shapes of
     a and b must be broadcastable. If two line segments are co-linear, we will always report them as non-intersecting
    Args:
        a: a floating point tensor of shape [..., 2, 2] representing a list of line segments from [..., 0, :] to
         [..., 1, :]
        b: a floating point tensor of shape [..., 2, 2] representing a list of line segments from [..., 0, :] to
         [..., 1, :]
        include_boundary: If true, line segments can intersect at their endpoints.

    Returns:
        (indicator, intersection)
        indicator: A boolean tensor of shape [..., ] that is true iff the two line segments intersect. The leading part
         of the shape will be the broadcast leading shapes of a and b
        intersection: A float tensor of shape [..., 2] that is equal to the intersection point where indicator is
         true, and an arbitrary value otherwise. The leading part of the shape will be the broadcast leading shapes
         of a and b.

    """
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)

    d = b[..., 0, :] - a[..., 0, :]
    l_a = a[..., 1, :] - a[..., 0, :]
    l_b = b[..., 1, :] - b[..., 0, :]

    alpha_a = d[..., 0] * l_b[..., 1] - d[..., 1] * l_b[..., 0]
    alpha_b = d[..., 0] * l_a[..., 1] - d[..., 1] * l_a[..., 0]

    delta = l_a[..., 0] * l_b[..., 1] - l_a[..., 1] * l_b[..., 0]

    if include_boundary:
        less_than = tf.math.less_equal
        sign_match = tf.logical_and(
            tf.logical_or(tf.sign(alpha_a) == tf.sign(delta), tf.sign(alpha_a) == 0.),
            tf.logical_or(tf.sign(alpha_b) == tf.sign(delta), tf.sign(alpha_b) == 0.))
    else:
        sign_match = tf.logical_and(tf.sign(alpha_a) == tf.sign(delta), tf.sign(alpha_b) == tf.sign(delta))
        less_than = tf.math.less

    a_in_bounds = tf.logical_and(less_than(tf.constant(0., alpha_a.dtype), tf.abs(alpha_a)),
                                 less_than(tf.abs(alpha_a), tf.abs(delta)))
    b_in_bounds = tf.logical_and(less_than(tf.constant(0., alpha_b.dtype), tf.abs(alpha_b)),
                                 less_than(tf.abs(alpha_b), tf.abs(delta)))
    both_in_bounds = tf.logical_and(a_in_bounds, b_in_bounds)

    indicator = tf.logical_and(sign_match, both_in_bounds)
    indicator = tf.logical_and(indicator, delta != 0.)

    t = tf.math.divide_no_nan(alpha_a, delta)

    intersection = a[..., 0, :] + t[..., None] * l_a

    return indicator, intersection


def point_in_polygon(polygon, point, include_boundary=True, check_ray=None):
    """

    Args:
        polygon: A [..., N, 2, 2] float tensor of directed polygon edges
        point: A [..., 2] float tensor of point coordinates
        include_boundary: If true, points are considered in the interior if they intersect a polygon segment

    Returns:
        A [...,] boolean tensor indicating if the point is in the interior of the polygon
    """

    polygon = tf.convert_to_tensor(polygon)
    point = tf.convert_to_tensor(point)
    if check_ray is None:
        check_ray = tf.constant((np.pi, np.exp(1.)), dtype=point.dtype)

    # we use a modified partial version of the line segment intersection algorithm above
    # to check each edge of the polygon for intersection with a horizontal ray
    # you can follow along with the above by analogy if you set a = point, b = polygon_edges

    d = polygon[..., 0, :] - point[..., None, :]
    l_a = check_ray
    l_b = polygon[..., 1, :] - polygon[..., 0, :]

    alpha_a = d[..., 0] * l_b[..., 1] - d[..., 1] * l_b[..., 0]
    alpha_b = d[..., 0] * l_a[..., 1] - d[..., 1] * l_a[..., 0]

    delta = l_a[..., 0] * l_b[..., 1] - l_a[..., 1] * l_b[..., 0]

    sign_match = tf.logical_and(
        tf.logical_or(tf.sign(alpha_a) == tf.sign(delta), tf.sign(alpha_a) == 0.),
        tf.logical_or(tf.sign(alpha_b) == tf.sign(delta), tf.sign(alpha_b) == 0.))

    b_interior = tf.logical_and(tf.math.less(tf.constant(0., alpha_b.dtype), tf.abs(alpha_b)),
                                tf.math.less(tf.abs(alpha_b), tf.abs(delta)))

    b_leading_edge = alpha_b == 0.

    b_in_bounds = tf.logical_or(b_interior, b_leading_edge)

    if include_boundary:
        a_in_bounds = tf.math.less_equal(tf.constant(0., alpha_a.dtype), tf.abs(alpha_a))
    else:
        a_in_bounds = tf.math.less(tf.constant(0., alpha_a.dtype), tf.abs(alpha_a))

    both_in_bounds = tf.logical_and(a_in_bounds, b_in_bounds)

    indicator = tf.logical_and(sign_match, both_in_bounds)

    indicator = tf.logical_and(indicator, delta != 0.)

    all_zero = tf.logical_and(alpha_a == alpha_b, tf.logical_and(alpha_b == delta, delta == 0.))
    horizontal_should_intercept = tf.logical_and(all_zero, d[..., 0] > 0.)

    indicator = tf.logical_or(indicator, horizontal_should_intercept)

    # now, we count the intersections:
    crossings = tf.reduce_sum(tf.cast(indicator, tf.int32), axis=-1)

    in_polygon = tf.math.mod(crossings, 2) == 1

    # brute force check all vertices against points
    matches_vertex = tf.reduce_all(polygon[..., 0, :] == point[..., None, :], axis=-1)
    matches_any_vertex = tf.reduce_any(matches_vertex, axis=-1)

    if include_boundary:
        in_polygon = tf.logical_or(in_polygon, matches_any_vertex)
    else:
        in_polygon = tf.logical_and(in_polygon, tf.logical_not(matches_any_vertex))

    return in_polygon
