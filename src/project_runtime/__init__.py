from functools import lru_cache
import warnings

import torch


@lru_cache(maxsize=None)
def _supports_fft(device):
    try:
        with torch.enable_grad():
            values = torch.ones((2, 2, 144), dtype=torch.float32, device=device, requires_grad=True)
            weight = torch.ones((2, 2, 16, 2), dtype=torch.float32, device=device, requires_grad=True)
            spectrum = torch.fft.rfft(values, dim=-1)
            transformed = spectrum.new_zeros((2, 2, spectrum.shape[-1]))
            transformed[..., :16] = torch.einsum(
                "bim,iom->bom", spectrum[..., :16], torch.view_as_complex(weight),
            )
            torch.fft.irfft(transformed, n=144, dim=-1).square().mean().backward()
        return True
    except (RuntimeError, NotImplementedError, TypeError):
        return False


def select_device(device="auto", *, require_fft=False):
    """Prefer CUDA (also ROCm), then Apple MPS, then CPU; allow an override."""
    automatic = device is None or str(device) == "auto"
    if automatic:
        if torch.cuda.is_available():
            selected = torch.device("cuda")
        elif torch.backends.mps.is_available():
            selected = torch.device("mps")
        else:
            selected = torch.device("cpu")
    else:
        selected = torch.device(device)
    if require_fft and selected.type == "mps" and not _supports_fft(str(selected)):
        if not automatic:
            raise RuntimeError("This MPS backend cannot run the FNO Fourier backward pass; use device='cpu'.")
        warnings.warn("MPS Fourier operations are unavailable; running the FNO on CPU.", stacklevel=2)
        return torch.device("cpu")
    return selected


def device_dtype(device, dtype):
    """Metal cannot allocate float64 or complex128 tensors."""
    if torch.device(device).type == "mps":
        return {torch.float64: torch.float32, torch.complex128: torch.complex64}.get(dtype, dtype)
    return dtype


def move_tensor(tensor, device, *, dtype=None):
    return tensor.to(device=device, dtype=device_dtype(device, tensor.dtype if dtype is None else dtype))


def prepare_model(model, device="auto"):
    selected = select_device(device, require_fft=getattr(model, "requires_fft", False))
    parameter = next(model.parameters())
    model.to(device=selected, dtype=device_dtype(selected, parameter.dtype))
    return next(model.parameters())


def make_generator(device, seed):
    # CPU-generated noise also supports MPS versions without device generators.
    generator_device = device if torch.device(device).type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def normal_noise(points, generator):
    noise = torch.randn(points.shape, generator=generator, dtype=points.dtype, device=generator.device)
    return noise.to(points.device)


def synchronize_device(device):
    """Wait for accelerator work before taking a wall-clock timestamp."""
    selected = torch.device(device)
    if selected.type == "cuda":
        torch.cuda.synchronize(selected)
    elif selected.type == "mps":
        torch.mps.synchronize()
