from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

import  warnings
warnings.filterwarnings(
    "ignore",
    message=r"The 'id' argument is deprecated and will be removed in v0\.46\.",
    category=qml.exceptions.PennyLaneDeprecationWarning,
    module=r"pennylane\.operation",
)
warnings.filterwarnings(
    "ignore",
    message=r"Using 'id' to add a custom label to your operator is deprecated\.",
    category=qml.exceptions.PennyLaneDeprecationWarning,
    module=r"pennylane\.operation",
)

GATE_MAP = {
    "rx": qml.RX,
    "rz": qml.RZ,
    "cz": qml.CZ,
    "rzz": qml.IsingZZ,
    "zz": qml.IsingZZ,
    "x": qml.PauliX,
    "sx": qml.SX,
    "id": qml.Identity,
    "identity": qml.Identity,
}

@dataclass
class LayerSpec:
    """Specification for a single ansatz layer."""

    name: str
    args: dict[str, Any]

    def __post_init__(self) -> None:
        if 'range' in self.args:
            if 'distance' not in self.args:
                self.args['distance'] = self.args.get('range', 1)
            self.args.pop('range', None)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable dictionary representation."""
        return {self.name: copy.deepcopy(self.args)}

    @classmethod
    def from_dict(cls, layer_dict: dict[str, Any]) -> "LayerSpec":
        """Create a LayerSpec from a dictionary representation."""

        if len(layer_dict) != 1:
            raise ValueError(f"Layer dictionary must contain exactly one key-value pair, but got {len(layer_dict)} pairs: {layer_dict}")

        name, args = next(iter(layer_dict.items()))
        return cls(name=name, args=copy.deepcopy(args))
    
# --- Gate Helper ---

def resolve_gate(gate: str | Any) -> Any:
    """Resolve a gate name to a PennyLane gate class."""

    if isinstance(gate, str):
        if gate not in GATE_MAP:
            raise ValueError(f"Unknown gate '{gate}'. Available gates: {list(GATE_MAP.keys())}")
        return GATE_MAP[gate]
    return gate

# --- Edge Helpers ---

def unique_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove duplicate edges from a list of tuples."""

    seen = set()
    unique = []

    for a, b in edges:
        if a == b:
            continue

        key = tuple(sorted((a, b)))

        if key not in seen:
            seen.add(key)
            unique.append((a, b))

    return unique

def get_ring_edges(n_qubits: int, distance: int = 1) -> list[tuple[int, int]]:
    """Generate a list of edges for a ring entanglement pattern."""

    if n_qubits < 2:
        return []

    if distance < 1:
        raise ValueError(f"Ring distance must be >= 1, got {distance}.")

    edges = [(i, (i + distance) % n_qubits) for i in range(n_qubits)]
    return unique_edges(edges)

def get_parallel_ring_edges(n_qubits: int, distance: int = 1) -> list[tuple[int, int]]:
    """Generate a list of edges for a parallel ring entanglement pattern."""

    if n_qubits < 2:
        return []

    edges = []
    for i in range(distance + 1):
        sublayer = [(i, (i + distance) % n_qubits) for i in range(i, n_qubits, distance + 1)]
        edges.extend(unique_edges(sublayer))

    return edges

# --- Feature-Index Helpers ---

def feature_indices(
    input_dim: int,
    n_qubits: int,
    upload_idx: int = 0,
    strategy: str = "same",
    wrap: bool = True,
) -> list[int]:
    """Generate a list of feature indices to upload to a layer of the ansatz."""

    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")

    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}.")

    if strategy == "same":
        start = 0
    elif strategy == "cyclic":
        start = upload_idx
    elif strategy == "block":
        start = upload_idx * n_qubits
    else:
        raise ValueError(f"Unknown feature strategy '{strategy}'. Supported strategies are: same, cyclic, block.")

    indices = [start + i for i in range(n_qubits)]

    if wrap:
        return [i % input_dim for i in indices]

    if max(indices) >= input_dim:
        raise ValueError(f"Feature indices {indices} exceed input_dim={input_dim}. Use wrap=True or change the strategy.")

    return indices

def used_features_from_layers(layers: list[LayerSpec]) -> set[int]:
    """Return all explicitly used feature indices from encoding layers."""

    used = set()

    for layer in layers:
        if layer.name in ENCODING_LAYERS:
            for f in layer.args.get("features", []):
                used.add(int(f))

    return used

def feature_coverage(layers: list[LayerSpec], input_dim: int) -> float:
    """Return the proportion of input features used by the ansatz."""

    if input_dim <= 0:
        return 0.0

    return len(used_features_from_layers(layers)) / input_dim

# --- Layer Functions ---

def angle_encoding(
    inputs: Any,
    weights: Any,
    idx: int,
    wires: list[int],
    *,
    gate: str | Any = "rz",
    features: list[int],
) -> int:
    """Apply a simple angle encoding layer to the active circuit."""

    gate = resolve_gate(gate)

    if gate.num_params == 0:
        raise ValueError("angle_encoding requires a parameterised gate.")

    if len(features) != len(wires):
        raise ValueError(f"angle_encoding expected {len(wires)} features, but got {len(features)}: {features}")

    f = 0
    for wire in wires:
        gate(inputs[..., features[f]], wires=wire, id=f"f{features[f]}")
        f += 1

    return idx

def linear_pairwise_encoding(
    inputs: Any,
    weights: Any,
    idx: int,
    wires: list[int],
    *,
    gate: str | Any = "rzz",
    features: list[int],
    distance: int = 1,
    include_single: bool = True,
    single_gate: str = "rz",
) -> int:
    """Apply a linear pairwise feature encoding layer to the active circuit."""

    n_qubits = len(wires)

    if len(features) != n_qubits:
        raise ValueError(f"linear_pairwise_encoding expected {n_qubits} feature indices, but got {len(features)}: {features}")

    pair_gate = resolve_gate(gate)
    single_gate = resolve_gate(single_gate)

    if pair_gate.num_params == 0:
        raise ValueError("linear_pairwise_encoding requires a parameterised two-qubit gate.")

    if include_single:
        if single_gate.num_params == 0:
            raise ValueError("single_gate in linear_pairwise_encoding must be parameterised if include_single=True.")

        for q, wire in enumerate(wires):
            single_gate(inputs[..., int(features[q])], wires=wire, id=f"f{features[q]}")

    edges = get_ring_edges(n_qubits=n_qubits, distance=distance)

    for a, b in edges:
        fa = int(features[a])
        fb = int(features[b])

        angle = inputs[..., fa] * inputs[..., fb]
        pair_gate(angle, wires=[wires[a], wires[b]])

    return idx

def parallel_pairwise_encoding(
    inputs: Any,
    weights: Any,
    idx: int,
    wires: list[int],
    *,
    gate: str | Any = "rzz",
    features: list[int],
    distance: int = 1,
    include_single: bool = True,
    single_gate: str = "rz",
) -> int:
    """Apply a parallel pairwise feature encoding layer to the active circuit."""

    n_qubits = len(wires)

    if len(features) != n_qubits:
        raise ValueError(f"parallel_pairwise_encoding expected {n_qubits} feature indices, but got {len(features)}: {features}")

    pair_gate = resolve_gate(gate)
    single_gate = resolve_gate(single_gate)

    if pair_gate.num_params == 0:
        raise ValueError("parallel_pairwise_encoding requires a parameterised two-qubit gate.")

    if include_single:
        if single_gate.num_params == 0:
            raise ValueError("single_gate in parallel_pairwise_encoding must be parameterised if include_single=True.")

        for q, wire in enumerate(wires):
            single_gate(inputs[..., int(features[q])], wires=wire, id=f"f{features[q]}")

    edges = get_parallel_ring_edges(n_qubits=n_qubits, distance=distance)

    for a, b in edges:
        fa = int(features[a])
        fb = int(features[b])

        angle = inputs[..., fa] * inputs[..., fb]
        pair_gate(angle, wires=[wires[a], wires[b]])

    return idx

def rotation_layer(
    inputs: Any,
    weights: Any,
    idx: int,
    wires: list[int],
    *,
    gate: str | Any = "rx",
) -> int:
    """Apply a rotation layer to the active circuit."""

    gate = resolve_gate(gate)

    if gate.num_params == 0:
        raise ValueError("rotation_layer requires a parameterised gate.")

    for wire in wires:
        params = weights[idx:idx + gate.num_params]
        gate(*params, wires=wire, id=f"w{idx}{f'-{idx + gate.num_params - 1}' if gate.num_params > 1 else ''}")
        idx += gate.num_params

    return idx

def linear_ring(
    inputs: Any,
    weights: Any,
    idx: int,
    wires: list[int],
    *,
    gate: str | Any = "cz",
    distance: int = 1,
) -> int:
    """Apply a linear ring entanglement layer to the active circuit."""

    gate = resolve_gate(gate)
    edges = get_ring_edges(n_qubits=len(wires), distance=distance)

    for a, b in edges:
        if gate.num_params == 0:
            gate(wires=[wires[a], wires[b]], id=f"e{a}-{b}")
        else:
            params = weights[idx:idx + gate.num_params]
            gate(*params, wires=[wires[a], wires[b]], id=f"w{idx}{f'-{idx + gate.num_params - 1}' if gate.num_params > 1 else ''}")
            idx += gate.num_params

    return idx

def parallel_ring(
    inputs: Any,
    weights: Any,
    idx: int,
    wires: list[int],
    *,
    gate: str | Any,
    distance: int = 1,
) -> int:
    """Apply a parallel ring entanglement layer to the active circuit."""

    gate = resolve_gate(gate)
    edges = get_parallel_ring_edges(n_qubits=len(wires), distance=distance)

    for a, b in edges:
        if gate.num_params == 0:
            gate(wires=[wires[a], wires[b]], id=f"e{a}-{b}")
        else:
            params = weights[idx:idx + gate.num_params]
            gate(*params, wires=[wires[a], wires[b]], id=f"w{idx}{f'-{idx + gate.num_params - 1}' if gate.num_params > 1 else ''}")
            idx += gate.num_params

    return idx

ENCODING_LAYERS = {
    "angle_encoding",
    "linear_pairwise_encoding",
    "parallel_pairwise_encoding",
}

TRAINABLE_LAYERS = {
    "rotation_layer",
    "linear_ring",
    "parallel_ring",
}

LAYER_FUNCTIONS: dict[str, Callable[..., int]] = {
    "angle_encoding": angle_encoding,
    "linear_pairwise_encoding": linear_pairwise_encoding,
    "parallel_pairwise_encoding": parallel_pairwise_encoding,
    "rotation_layer": rotation_layer,
    "linear_ring": linear_ring,
    "parallel_ring": parallel_ring,
}

# --- Parameter Counting ---

def count_angle_encoding_params(
    n_qubits: int,
    *,
    gate: str | Any,
    features: list[int],
) -> int:
    return 0

def count_pairwise_encoding_params(
    n_qubits: int,
    *,
    features: list[int],
    gate: str | Any = "rzz",
    distance: int = 1,
    include_single: bool = True,
    single_gate: str | Any = "rz",
) -> int:
    return 0

def count_rotation_layer_params(
    n_qubits: int,
    *,
    gate: str | Any,
) -> int:
    resolved_gate = resolve_gate(gate)
    return n_qubits * resolved_gate.num_params

def count_linear_ring_params(
    n_qubits: int,
    *,
    gate: str | Any,
    distance: int = 1,
) -> int:
    gate = resolve_gate(gate)
    edges = get_ring_edges(n_qubits=n_qubits, distance=distance)
    return len(edges) * gate.num_params

def count_parallel_ring_params(
    n_qubits: int,
    *,
    gate: str | Any,
    distance: int = 1,
) -> int:
    gate = resolve_gate(gate)
    edges = get_parallel_ring_edges(n_qubits=n_qubits, distance=distance)
    return len(edges) * gate.num_params

LAYER_PARAM_COUNTERS: dict[str, Callable[..., int]] = {
    "angle_encoding": count_angle_encoding_params,
    "linear_pairwise_encoding": count_pairwise_encoding_params,
    "parallel_pairwise_encoding": count_pairwise_encoding_params,
    "rotation_layer": count_rotation_layer_params,
    "linear_ring": count_linear_ring_params,
    "parallel_ring": count_parallel_ring_params,
}

def count_layer_params(layer: LayerSpec, n_qubits: int) -> int:
    """Count trainable parameters for a single ansatz layer."""

    if layer.name not in LAYER_PARAM_COUNTERS:
        raise ValueError(f"No parameter counter registered for layer '{layer.name}'. Available counters: {list(LAYER_PARAM_COUNTERS.keys())}" )

    return LAYER_PARAM_COUNTERS[layer.name](
        n_qubits=n_qubits,
        **layer.args,
    )

def count_total_params(layers: list[LayerSpec], n_qubits: int) -> int:
    """Count all trainable parameters for a layered ansatz."""
    return sum(count_layer_params(layer, n_qubits) for layer in layers)

# --- AnsatzSpec ---

@dataclass
class AnsatzSpec:
    name: str
    layers: list[LayerSpec]
    wires: list[int]
    input_dim: int
    measurement_wires: list[int] | None = None

    def __post_init__(self) -> None:
        self.layers = copy.deepcopy(self.layers)

        upload_idx = 0
        for layer in self.layers:
            if layer.name not in LAYER_FUNCTIONS:
                raise ValueError(f"Unknown layer '{layer.name}'. Available layers: {list(LAYER_FUNCTIONS.keys())}")

            if layer.name in ENCODING_LAYERS:
                if "features" not in layer.args:
                    strategy = layer.args.pop("feature_strategy", "same")
                    wrap = bool(layer.args.pop("wrap", True))

                    layer.args["features"] = feature_indices(
                        input_dim=self.input_dim,
                        n_qubits=len(self.wires),
                        upload_idx=upload_idx,
                        strategy=strategy,
                        wrap=wrap,
                    )
                else:
                    if len(layer.args["features"]) != len(self.wires):
                        raise ValueError(f"{layer.name} expected {len(self.wires)} features, but got {len(layer.args['features'])}: {layer.args['features']}")
                upload_idx += 1

        if self.measurement_wires is None:
            self.measurement_wires = list(self.wires)
        else:
            self.measurement_wires = self.measurement_wires

    @property
    def weight_shapes(self) -> dict[str, tuple[int]]:
        """Weight shapes compatible with qml.qnn.TorchLayer."""
        return {
            "weights": (
                count_total_params(self.layers, n_qubits=len(self.wires)),
            )
        }

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return self.weight_shapes["weights"][0]

    @property
    def n_features(self) -> int:
        """Number of input features expected by the QNode."""
        return self.input_dim

    @property
    def used_features(self) -> list[int]:
        """Sorted list of feature indices used by this ansatz."""
        return sorted(used_features_from_layers(self.layers))

    @property
    def feature_coverage(self) -> float:
        """Fraction of input features used by this ansatz."""
        return feature_coverage(self.layers, self.input_dim)

    def to_config(self) -> dict[str, Any]:
        """Return a serialisable dictionary representation."""
        return {
            "name": self.name,
            "layers": [layer.to_dict() for layer in self.layers],
            "wires": list(self.wires),
            "input_dim": self.input_dim,
            "measurement_wires": list(self.measurement_wires)
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        name: str | None = None,
        measurement_wires: list[int] | None = None,
    ) -> AnsatzSpec:
        """Create an AnsatzSpec from a dictionary configuration."""

        if "layers" not in config:
            raise ValueError("Ansatz config must contain 'layers'.")
        
        if "wires" not in config:
            raise ValueError("Ansatz config must contain 'wires'.")

        if "input_dim" not in config:
            raise ValueError("Ansatz config must contain 'input_dim'.")

        return cls(
            name=name or config.get("name", "unnamed_ansatz"),
            layers=[LayerSpec.from_dict(layer_config) for layer_config in config["layers"]],
            wires=list(config["wires"]),
            input_dim=int(config["input_dim"]),
            measurement_wires=(
                measurement_wires
                if measurement_wires is not None
                else config.get("measurement_wires", None)
            ),
        )

    def summary(self) -> dict[str, Any]:
        """Return useful experiment metadata."""
        return {
            "name": self.name,
            "n_layers": len(self.layers),
            "layer_sequence": [layer.name for layer in self.layers],
            "trainable_params": self.n_params,
            "weight_shapes": self.weight_shapes,
            "input_dim": self.input_dim,
            "used_features": self.used_features,
            "n_used_features": len(self.used_features),
            "feature_coverage": self.feature_coverage,
            "wires": list(self.wires),
            "measurement_wires": list(self.measurement_wires),
        }

    def apply_ansatz(
        self,
        inputs: Any,
        weights: Any
    ) -> None:
        """Apply the layered ansatz to the active circuit."""

        idx = 0
        for layer in self.layers:
            idx = LAYER_FUNCTIONS[layer.name](
                inputs=inputs,
                weights=weights,
                idx=idx,
                wires=self.wires,
                **layer.args,
            )

        expected = count_total_params(self.layers, n_qubits=len(self.wires))
        if idx != expected:
            raise RuntimeError(
                f"Ansatz consumed {idx} parameters, but expected {expected}."
            )

    def build_qnode(
        self,
        device_name: str = "default.qubit",
        interface: str = "torch",
        diff_method: str = "best",
        shots: int | None = None,
        device_kwargs: dict[str, Any] | None = None,
    ):
        """Build a PennyLane QNode from this ansatz."""

        device_kwargs = dict(device_kwargs or {})
        device_kwargs["wires"] = self.wires

        if shots is not None:
            device_kwargs["shots"] = shots

        dev = qml.device(device_name, **device_kwargs)

        qnode_kwargs: dict[str, Any] = {
            "interface": interface,
        }

        if diff_method is not None:
            qnode_kwargs["diff_method"] = diff_method

        @qml.qnode(dev, **qnode_kwargs)
        def circuit(inputs, weights):
            self.apply_ansatz(
                inputs=inputs,
                weights=weights
            )

            return [qml.probs(wires=self.measurement_wires)]

        return circuit

    def build_qlayer(
        self,
        device_name: str = "default.qubit",
        interface: str = "torch",
        shots: int | None = None,
        diff_method: str = "best",
        device_kwargs: dict[str, Any] | None = None,
    ):
        """Build a PennyLane TorchLayer from this ansatz."""

        circuit = self.build_qnode(
            device_name=device_name,
            interface=interface,
            shots=shots,
            diff_method=diff_method,
            device_kwargs=device_kwargs,
        )

        qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

        return qlayer, circuit, self.weight_shapes, self.n_features

    def draw_text(
        self,
        decimals: int = 2,
        interface: str = "autograd",
    ) -> str:
        """Return qml.draw text output for quick inspection."""

        circuit = self.build_qnode(
            device_name="default.qubit",
            interface=interface,
        )

        x = np.zeros(self.input_dim)
        weights = np.zeros(self.n_params)

        return qml.draw(circuit, decimals=decimals)(x, weights)

    def draw_mpl(
        self,
        decimals: int | None = None,
        interface: str = "autograd",
    ):
        """Return qml.draw_mpl figure and axis."""

        circuit = self.build_qnode(
            device_name="default.qubit",
            interface=interface,
        )

        x = np.zeros(self.input_dim)
        weights = np.zeros(self.n_params)

        drawer = qml.draw_mpl(circuit, decimals=decimals)(x, weights)
        plt.show()
        return drawer

# --- Template Loading ---

def load_ansatz_templates(template_path: str | Path = "ansatze.json") -> dict[str, Any]:
    """Load ansatz templates from a JSON file."""

    path = Path(template_path)

    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        templates = json.load(f)

    if not isinstance(templates, dict):
        raise ValueError(
            f"Template file must contain a dictionary of named templates, "
            f"got {type(templates)}."
        )

    return templates

def resolve_layer_template(
    template: str | list[dict[str, Any]] | dict[str, Any],
    template_path: str | Path = "ansatze.json",
) -> tuple[str, list[dict[str, Any]]]:
    """Resolve a template name or inline layer list into a list of layer dictionaries."""

    if isinstance(template, str):
        templates = load_ansatz_templates(template_path)

        if template not in templates:
            raise ValueError(
                f"Unknown ansatz template '{template}'. "
                f"Available templates: {list(templates.keys())}"
            )

        raw_layers = templates[template]

        if not isinstance(raw_layers, list):
            raise ValueError(
                f"Template '{template}' must be a list of layers, "
                f"got {type(raw_layers)}."
            )

        return template, copy.deepcopy(raw_layers)

    if isinstance(template, list):
        return "inline_ansatz", copy.deepcopy(template)

    if isinstance(template, dict):
        if "layers" in template:
            name = template.get("name", "inline_ansatz")
            raw_layers = template["layers"]

            if not isinstance(raw_layers, list):
                raise ValueError(
                    f"Inline template 'layers' must be a list, "
                    f"got {type(raw_layers)}."
                )

            return name, copy.deepcopy(raw_layers)

        if len(template) == 1:
            name, raw_layers = next(iter(template.items()))

            if not isinstance(raw_layers, list):
                raise ValueError(
                    f"Inline template '{name}' must map to a list of layers, "
                    f"got {type(raw_layers)}."
                )

            return str(name), copy.deepcopy(raw_layers)

    raise TypeError(
        "template must be either a template name, a list of layer dictionaries, "
        "or a dictionary containing a 'layers' field."
    )

def resolve_ansatz_spec(
    template: str | list[dict[str, Any]] | dict[str, Any],
    *,
    wires: list[int],
    input_dim: int,
    template_path: str | Path = "ansatze.json",
    measurement_wires: list[int] | None = None,
    name: str | None = None,
) -> AnsatzSpec:
    """Resolve an ansatz template into a full AnsatzSpec."""

    resolved_name, raw_layers = resolve_layer_template(
        template=template,
        template_path=template_path,
    )

    layer_specs = [
        LayerSpec.from_dict(layer_config)
        for layer_config in raw_layers
    ]

    return AnsatzSpec(
        name=name or resolved_name,
        layers=layer_specs,
        wires=wires,
        input_dim=input_dim,
        measurement_wires=measurement_wires,
    )

if __name__ == "__main__":
    # Example usage
    ansatz = resolve_ansatz_spec(
        template="hea_rzz_ring",
        wires=[0, 1, 2],
        input_dim=4,
    )

    print("Ansatz Summary:")
    print(ansatz.summary())

    ansatz.draw_mpl()