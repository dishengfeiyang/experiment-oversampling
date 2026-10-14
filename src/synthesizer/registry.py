from .gaussian import GaussianSynthesizer


def build_synthesizer(name, **kwargs):
    if name == "gaussian":
        return GaussianSynthesizer(**kwargs)
    else:
        raise ValueError(f"Unknown synthesizer: {name}")