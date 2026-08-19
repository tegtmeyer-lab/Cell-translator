import pytest

tf = pytest.importorskip("tensorflow")
from cell_translator.models.cyclegan import build_generator, build_discriminator


def test_cyclegan_shapes():
    g = build_generator(16, 16, "G_test")
    d = build_discriminator(16, "D_test")
    x = tf.zeros((3, 16), dtype=tf.float32)
    assert tuple(g(x).shape) == (3, 16)
    assert tuple(d(x).shape) == (3, 1)
