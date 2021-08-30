import pytest
from pytest import approx

import tensorflow as tf

from tf_polygon.primitives import signed_point_line_area, signed_point_line_distance

unit_x_line_segment = tf.convert_to_tensor(((0., 0.),
                                            (1., 0.)))
unit_y_line_segment = tf.convert_to_tensor(((0., 0.),
                                            (0., 1.)))


def test_point_on_line_distance_zero():
    assert signed_point_line_distance(unit_x_line_segment, (0., 0.)).numpy() == approx(0.)
    assert signed_point_line_distance(unit_x_line_segment, (.5, 0.)).numpy() == approx(0.)
    assert signed_point_line_distance(unit_x_line_segment, (1., 0.)).numpy() == approx(0.)
    assert signed_point_line_distance(unit_x_line_segment, (-1., 0.)).numpy() == approx(0.)
    assert signed_point_line_distance(unit_x_line_segment, (2., 0.)).numpy() == approx(0.)

    assert signed_point_line_distance(unit_y_line_segment, (0., 0.)).numpy() == approx(0.)
    assert signed_point_line_distance(unit_y_line_segment, (0., .5)).numpy() == approx(0.)
    assert signed_point_line_distance(unit_y_line_segment, (0., 1.)).numpy() == approx(0.)
    assert signed_point_line_distance(unit_y_line_segment, (0., -1.)).numpy() == approx(0.)
    assert signed_point_line_distance(unit_y_line_segment, (0., 2.)).numpy() == approx(0.)


def test_unit_distance():
    assert signed_point_line_distance(unit_x_line_segment, (0., 1.)).numpy() == approx(1.)
    assert signed_point_line_distance(unit_x_line_segment, (.5, 1.)).numpy() == approx(1.)
    assert signed_point_line_distance(unit_x_line_segment, (1., 1.)).numpy() == approx(1.)
    assert signed_point_line_distance(unit_x_line_segment, (0., -1.)).numpy() == approx(-1.)
    assert signed_point_line_distance(unit_x_line_segment, (.5, -1.)).numpy() == approx(-1.)
    assert signed_point_line_distance(unit_x_line_segment, (1., -1.)).numpy() == approx(-1.)

    assert signed_point_line_distance(unit_y_line_segment, (1., 0.,)).numpy() == approx(-1.)
    assert signed_point_line_distance(unit_y_line_segment, (1., .5,)).numpy() == approx(-1.)
    assert signed_point_line_distance(unit_y_line_segment, (1., 1.,)).numpy() == approx(-1.)
    assert signed_point_line_distance(unit_y_line_segment, (-1., 0.,)).numpy() == approx(1.)
    assert signed_point_line_distance(unit_y_line_segment, (-1., .5,)).numpy() == approx(1.)
    assert signed_point_line_distance(unit_y_line_segment, (-1., 1.,)).numpy() == approx(1.)


def test_zero_length_segment_has_zero_area():
    assert signed_point_line_area(((0., 0.), (0., 0.)), (1., 1.)).numpy() == approx(0.)


def test_derivative():
    point = tf.convert_to_tensor((1., 0.))

    with tf.GradientTape() as tape:
        tape.watch(point)
        d = signed_point_line_distance(unit_x_line_segment, point)

    grad = tape.gradient(d, point)

    assert tf.math.reduce_euclidean_norm(grad) > 0.


def test_broadcast():
    lines = tf.stack([unit_x_line_segment, unit_y_line_segment])
    points = tf.convert_to_tensor(((.5, .5), (-1., 1.)))

    d = signed_point_line_distance(lines[:, None, :, :], points[None, :, :])

    assert tf.reduce_all(tf.shape(d) == (2, 2))
