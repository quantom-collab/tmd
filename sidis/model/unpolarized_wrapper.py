"""
Simple unpolarized wrapper around TrainableModel.

This exposes a plain function-style API:
    differential_xsec = f(events, parameters)
where `parameters` can be:
    - a dict keyed by model parameter names, or
    - a flat vector in `parameter_names` order.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

try:
    from . import TrainableModel
except ImportError:
    # Support direct execution: python3 sidis/model/unpolarized_wrapper.py
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from sidis.model import TrainableModel

TensorLike = Union[torch.Tensor, Sequence[float]]
ParameterInput = Union[Mapping[str, Union[float, TensorLike]], TensorLike]


class UnpolarizedSIDISWrapper:
    """Wrapper for unpolarized SIDIS model evaluation."""

    def __init__(
        self,
        fnp_config: str = "fNPconfig_simple.yaml",
        expt_setup: Tuple[str, str] = ("p", "pi_plus"),
        rs: float = 140.0,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.expt_setup = list(expt_setup)
        self.rs = rs
        self.model = TrainableModel(
            fnp_config=fnp_config,
            experimental_target_fragmented_hadron=[self.expt_setup],
        )

        if getattr(self.model.qcf0, "sivers_flag", False):
            raise ValueError(
                "UnpolarizedSIDISWrapper requires an unpolarized card/config "
                "(sivers_flag must be False)."
            )

        if device is not None:
            self.model = self.model.to(device=device)
        if dtype is not None:
            self.model = self.model.to(dtype=dtype)

        self.model.eval()
        self.parameter_names = [
            name for name, param in self.model.named_parameters() if param.requires_grad
        ]

    def set_parameters(self, parameters: ParameterInput) -> None:
        """Update trainable model parameters.

        Supported formats:
        - dict: {"module.path.param_name": value_or_tensor, ...}
        - flat vector/list/tensor in `self.parameter_names` order
        """
        named_params: Dict[str, torch.nn.Parameter] = dict(self.model.named_parameters())

        with torch.no_grad():
            if isinstance(parameters, Mapping):
                for name, value in parameters.items():
                    if name not in named_params:
                        raise KeyError(f"Unknown parameter name: {name}")
                    target = named_params[name]
                    value_tensor = torch.as_tensor(
                        value, dtype=target.dtype, device=target.device
                    ).reshape_as(target)
                    target.copy_(value_tensor)
                return

            flat_values = torch.as_tensor(parameters, dtype=torch.get_default_dtype())
            if flat_values.numel() != len(self.parameter_names):
                raise ValueError(
                    "Flat parameter vector length mismatch: "
                    f"got {flat_values.numel()}, expected {len(self.parameter_names)}."
                )
            flat_values = flat_values.reshape(-1)

            for idx, name in enumerate(self.parameter_names):
                target = named_params[name]
                value_tensor = flat_values[idx].to(device=target.device, dtype=target.dtype)
                target.copy_(value_tensor.reshape_as(target))

    def predict(
        self,
        events: TensorLike,
        parameters: Optional[ParameterInput] = None,
        expt_setup: Optional[Tuple[str, str]] = None,
        rs: Optional[float] = None,
    ) -> torch.Tensor:
        """Evaluate differential cross sections for input events.

        Event columns follow TrainableModel.forward:
            [x, PhT, Q, z] (+ optional angle columns ignored for unpolarized cards)
        """
        if parameters is not None:
            self.set_parameters(parameters)

        events_tensor = torch.as_tensor(
            events,
            dtype=next(self.model.parameters()).dtype,
            device=next(self.model.parameters()).device,
        )
        if events_tensor.ndim != 2 or events_tensor.shape[1] < 4:
            raise ValueError("events must have shape (n_events, >=4)")

        active_setup = list(expt_setup) if expt_setup is not None else self.expt_setup
        active_rs = self.rs if rs is None else rs
        with torch.no_grad():
            return self.model(events_tensor, expt_setup=active_setup, rs=active_rs)


def main() -> torch.Tensor:
    wrapper = UnpolarizedSIDISWrapper(fnp_config="fNPconfig_simple.yaml")

    events = torch.tensor(
        [
            [0.1, 0.5, 20.0, 0.3],
            [0.2, 0.7, 30.0, 0.4],
        ],
        dtype=torch.float64,
    )

    print("Parameter names (trainable order):")
    print(wrapper.parameter_names)

    # flat numeric values (for scalar params)
    values = [p.detach().cpu().item() for _, p in wrapper.model.named_parameters() if p.numel() == 1]
    print('values:')
    print(values)

    xsec = wrapper.predict(events, parameters=values)
    print("\nPredicted differential cross sections:")
    print(xsec)
    print(f"\nOutput shape: {tuple(xsec.shape)}")
    return xsec


if __name__ == "__main__":
    _ = main()
