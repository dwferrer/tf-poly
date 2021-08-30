import pytest

import tensorflow as tf

from tf_polygon.primitives import point_in_polygon, get_edges

unit_square_vertices = tf.convert_to_tensor((
    (0., 0.),
    (0., 1.),
    (1., 1.),
    (1., 0.),
))

unit_square = get_edges(unit_square_vertices)


def test_point_in_unit_square():
    assert point_in_polygon(unit_square, (.5, .5))


def test_points_out_of_unit_square():
    assert not point_in_polygon(unit_square, (.5, 1.5))
    assert not point_in_polygon(unit_square, (.5, -1.5))
    assert not point_in_polygon(unit_square, (1.5, .5))
    assert not point_in_polygon(unit_square, (1.5, .5))

    assert not point_in_polygon(unit_square, (1.5, 1.5))
    assert not point_in_polygon(unit_square, (-1.5, 1.5))
    assert not point_in_polygon(unit_square, (-1.5, -1.5))
    assert not point_in_polygon(unit_square, (1.5, -1.5))


def test_exterior_points_colinear_with_edges():
    assert not point_in_polygon(unit_square, (1.5, 0.))
    assert not point_in_polygon(unit_square, (-.5, 0.))

    assert not point_in_polygon(unit_square, (1.5, 1.))
    assert not point_in_polygon(unit_square, (-.5, 1.))

    assert not point_in_polygon(unit_square, (0., 1.5))
    assert not point_in_polygon(unit_square, (0., -.5))

    assert not point_in_polygon(unit_square, (1., 1.5))
    assert not point_in_polygon(unit_square, (1., -.5))

    assert not point_in_polygon(unit_square, (1.5, 0.), include_boundary=False)
    assert not point_in_polygon(unit_square, (-.5, 0.), include_boundary=False)

    assert not point_in_polygon(unit_square, (1.5, 1.), include_boundary=False)
    assert not point_in_polygon(unit_square, (-.5, 1.), include_boundary=False)

    assert not point_in_polygon(unit_square, (0., 1.5), include_boundary=False)
    assert not point_in_polygon(unit_square, (0., -.5), include_boundary=False)

    assert not point_in_polygon(unit_square, (1., 1.5), include_boundary=False)
    assert not point_in_polygon(unit_square, (1., -.5), include_boundary=False)


def test_vertices_in_boundary():
    in_boundary = point_in_polygon(unit_square[None, :, :, :], unit_square_vertices)

    assert tf.shape(in_boundary) == 4
    assert tf.reduce_all(in_boundary)

    in_boundary = point_in_polygon(unit_square[None, :, :, :], unit_square_vertices, include_boundary=False)

    assert not tf.reduce_any(in_boundary)


def test_interior_check_ray_through_vertex():
    # NOTE: This and all the other check-ray tests are not actually checking the true ray direction
    #  now that we switched it from being horizontal to being transcendental
    assert point_in_polygon(
        get_edges((
            (-1., 0.),
            (0., 1.),
            (1., 0),
            (0., -1.)
        )),
        (0., 0.)
    )

    assert point_in_polygon(
        get_edges((
            (-1., 0.),
            (0., 1.),
            (1., 0),
            (0., -1.)
        )),
        (0., 0.),
        include_boundary=False
    )


def test_exterior_check_ray_through_vertex():
    assert not point_in_polygon(
        get_edges((
            (-1., 0.),
            (0., 1.),
            (1., 0),
            (0., -1.)
        )),
        (-1.5, 0.)
    )

    assert not point_in_polygon(
        get_edges((
            (-1., 0.),
            (0., 1.),
            (1., 0),
            (0., -1.)
        )),
        (-1.5, 0.),
        include_boundary=False
    )


def test_exterior_check_ray_through_single_vertex():
    assert not point_in_polygon(
        get_edges((
            (-1., 0.),
            (0., 1.),
            (1., 0),
            (0., -1.)
        )),
        (-1.5, 1.)
    )

    assert not point_in_polygon(
        get_edges((
            (-1., 0.),
            (0., 1.),
            (1., 0),
            (0., -1.)
        )),
        (-1.5, 1.),
        include_boundary=False,
    )
