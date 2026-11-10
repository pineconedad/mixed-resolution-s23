def load_adapter(model, checkpoint, device):
    if model == "pi3":
        from .pi3 import load_pi3
        return load_pi3(checkpoint, device)
    if model == "vggt":
        from .vggt import load_vggt
        return load_vggt(checkpoint, device)
    raise ValueError("Unsupported model; no checkpoint choice is inferred")
