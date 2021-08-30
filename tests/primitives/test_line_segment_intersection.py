import pytest

import tensorflow as tf

from tf_polygon.primitives import line_segment_intersection


def test_interior():
    indicator, intersect = line_segment_intersection(
        ((.5, 0.), (.5, 1.)),
        ((0., .5), (1., .5))
    )

    assert indicator
    assert tf.reduce_all(intersect == (.5, .5))


def test_boundary():
    indicator, intersect = line_segment_intersection(
        ((0., 0.), (1., 0.)),
        ((0., -1.), (0., 1.)),
        include_boundary=True,
    )
    assert indicator
    assert tf.reduce_all(intersect == (0., 0.))

    indicator, intersect = line_segment_intersection(
        ((0., 0.), (1., 0.)),
        ((1., -1.), (1., 1.)),
        include_boundary=True,
    )
    assert indicator
    assert tf.reduce_all(intersect == (1., 0.))

    indicator, intersect = line_segment_intersection(
        ((0., 0.), (1., 0.)),
        ((0., -1.), (0., 1.)),
        include_boundary=False,
    )
    assert not indicator

    indicator, intersect = line_segment_intersection(
        ((0., 0.), (1., 0.)),
        ((1., -1.), (1., 1.)),
        include_boundary=False,
    )
    assert not indicator


def test_non_intersecting():
    indicator, intersect = line_segment_intersection(
        ((.5, 0.), (.5, 1.)),
        ((0., -.5), (1., -.5))
    )

    assert not indicator


def test_derivative():
    a = tf.convert_to_tensor(((.5, 0.), (.5, 1.)))
    b = tf.convert_to_tensor(((0., .5), (1., .5)))

    with tf.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)

        indicator, intersect = line_segment_intersection(a, b)

    d_a, d_b = tape.gradient(intersect, (a, b))

    assert tf.math.reduce_euclidean_norm(d_a) > 0.
    assert tf.math.reduce_euclidean_norm(d_b) > 0.


def test_broadcast():
    a = tf.convert_to_tensor((((.5, 0.), (.5, 1.)),
                              ((0., .5), (1., .5))))

    b = tf.convert_to_tensor((((0., -.5), (1., -.5)),
                              ((1., -.5), (1., -1.5))))

    indicator, intersect = line_segment_intersection(a[:, None, :, :], b[None, :, :, :])

    assert tf.reduce_all(tf.shape(indicator) == (2, 2))
    assert tf.reduce_all(tf.shape(intersect) == (2, 2, 2))
