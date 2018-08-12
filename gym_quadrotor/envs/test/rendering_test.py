import pytest
import mock
import gym_quadrotor
import sys

# try to load pyglet, but in case we can't for whatever reason, just mock it at the module level
try:
    import pyglet
except:
    sys.modules["pyglet"] = mock.Mock()

from gym_quadrotor.envs.rendering import *


@pytest.fixture()
def renderer():
    renderer = Renderer()
    with mock.patch('gym_quadrotor.envs.rendering.rendering') as rendering_mock:
        def new_viewer(*args):
            return mock.Mock()
        rendering_mock.Viewer = mock.Mock(side_effect=new_viewer)
        yield renderer


def test_renderer_center(renderer):
    assert renderer.center is None

    # first center is directly set
    renderer.set_center(1.0)
    assert renderer.center == pytest.approx(1)

    # second update is smoothed
    renderer.set_center(2.0)
    assert renderer.center == pytest.approx(1.1)

    # after a reset, we again set directly
    renderer.set_center(None)
    assert renderer.center is None
    renderer.set_center(2.0)
    assert renderer.center == pytest.approx(2)

    # check that viewer is correctly recentered
    renderer.setup()
    renderer.set_center(2.0)

    renderer.viewer.set_bounds.assert_called_once_with(-5, 9, -1, 13)


def test_renderer_setup(renderer):
    renderer.setup()
    gym_quadrotor.envs.rendering.rendering.Viewer.assert_called_once()

    # second call to setup does nothing
    renderer.setup()
    gym_quadrotor.envs.rendering.rendering.Viewer.assert_called_once()


def test_renderer_close(renderer):
    # this is a no-op
    renderer.close()

    renderer.setup()
    # we need to cache the viewer here, because close will reset this variable to None
    viewer = renderer.viewer  # type: mock.Mock
    renderer.close()

    viewer.close.assert_called_once()
    assert renderer.viewer is None

    # closing via render(close=True) call should have the same effect
    renderer.setup()
    viewer = renderer.viewer  # type: mock.Mock
    renderer.render(close=True)
    viewer.close.assert_called_once()


def test_renderer_render(renderer):
    obj1 = mock.Mock()
    obj2 = mock.Mock()

    renderer.add_object(obj1)
    renderer.add_object(obj2)

    renderer.render()

    obj1.draw.assert_called_once()
    obj2.draw.assert_called_once()

    renderer.viewer.render.assert_called_once()


def test_renderer_line_drawing(renderer):
    renderer.setup()

    renderer.draw_line_2d(1, 2)
    renderer.viewer.draw_line.assert_called_once_with(1, 2)

    renderer.draw_line_3d((1, 2, 3), (4, 5, 6))
    renderer.viewer.draw_line.assert_called_with((1, 3), (4, 6))
