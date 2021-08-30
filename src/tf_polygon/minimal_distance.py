import tensorflow as tf

from tf_polygon.primitives import get_edges, point_in_polygon, point_line_segment_distance


def minimal_distance(poly_a, poly_b):
    x_a = tf.convert_to_tensor(poly_a)
    x_b = tf.convert_to_tensor(poly_b)

    e_a = get_edges(poly_a)
    e_b = get_edges(poly_b)

    a_in_b = point_in_polygon(e_b[..., None, :, :, :], x_a)
    a_in_b = tf.reduce_any(a_in_b, axis=-1)

    b_in_a = point_in_polygon(e_a[..., None, :, :, :], x_b)
    b_in_a = tf.reduce_any(b_in_a, axis=-1)

    intersection = tf.logical_or(a_in_b, b_in_a)

    # the minimal distance must occur between an edge of a and vertex of b or visa-versa
    d_a_b = point_line_segment_distance(e_a[..., :, None, :, :], x_b[..., None, :])
    d_a_b = tf.reduce_min(d_a_b, axis=[-2, -1])

    d_b_a = point_line_segment_distance(e_b[..., :, None, :, :], x_a[..., None, :])
    d_b_a = tf.reduce_min(d_b_a, axis=[-2, -1])

    d = tf.minimum(d_a_b, d_b_a)
    d = tf.where(intersection, tf.constant(0., d.dtype), d)

    return d



