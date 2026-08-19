"""Latent loading, translation, decoding, and artifact saving."""

__all__ = ["run_translation", "load_state_pair_latents", "translate_and_decode_source"]


def __getattr__(name):
    if name in set(__all__):
        from . import translate
        return getattr(translate, name)
    raise AttributeError(name)
