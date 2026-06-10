import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from pennylane import numpy as np
from scipy.linalg import hadamard
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pandas import DataFrame, Series
from random import choices, sample
import  warnings
warnings.filterwarnings(
    "ignore",
    message=r"The 'id' argument is deprecated and will be removed in v0\.46\.",
    category=qml.exceptions.PennyLaneDeprecationWarning,
    module=r"pennylane\.operation",
)

class ValReport(dict):
    def __str__(self):
        report_str = ""
        for label, metrics in self.items():
            report_str += f"{label}:\n"
            for metric, value in metrics.items():
                report_str += f"  {metric}: {value:.4f}\n"
        return report_str

class TimeInt(int):
    def __str__(self):
        secs, mins, hours = self, 0, 0
        if secs >= 60:
            mins = secs // 60
            secs = secs % 60
            if mins >= 60:
                hours = mins // 60
                mins = mins % 60
        return f"{f'{int(hours)}h ' if hours > 0 else ''}{f'{int(mins)}m ' if mins > 0 else ''}{secs:.2f}s"

def build_ecoc_matrix(n_classes: int, n_learners: int) -> np.ndarray:
    n = 2 ** int(np.ceil(np.log2(n_learners)) + 1)

    H = hadamard(n)[:n_classes]

    code = ((H + 1) // 2).astype(int)

    mask = (code.sum(0) > 0) & (code.sum(0) < n_classes)
    code = code[:, mask]

    ecoc = np.array(code[:, :n_learners]).T

    inc = -1
    for i in range(len(ecoc)):
        if np.array_equal(ecoc[i], ecoc[0]):
            inc += 1
        ecoc[i] = np.roll(ecoc[i], inc)
    return ecoc

def reuploading_qlayer(
    n_qubits: int,
    feats_per_qubit: int,
    device: qml.Device,
    reuploads: int = 2,
    entangle_between_reuploads: bool = True,
    trainable_layers: int | list[int] = None,
    ranges: int | list[int] | list[list[int]] = 1,
    rot_gates: list[type[qml.RX]] = None,
    ent_gate: type[qml.CZ] = qml.CZ,
    measurement_wires: list[int] = None,
):
    """
    Dynamic feature-reuploading quantum layer.

    Args:
        n_qubits: Number of qubits in the circuit.
        feats_per_qubit: Number of classical features loaded onto each qubit per reupload.
        reuploads: Number of times the feature loading and entangling blocks are repeated.
        entangle_between_reuploads: Whether to apply entanglement between feature loading blocks. If False, entanglement is only applied after the last feature loading block.
        trainable_layers: Number of trainable layers of Rot gates to apply after each feature loading block. Can also be a list specifying the number of trainable layers for each reupload.
        ranges: Range(s) for the strongly entangling pattern. Can be a single int applied to all reuploads, a list of ints specifying the range for each reupload, or a list of lists specifying the range for each layer within each reupload.
        rot_gates: List of single-qubit rotation gate types to use for feature encoding and trainable layers.
        ent_gate: The two-qubit gate used for entanglement (e.g., qml.CNOT, qml.CZ).
        measurement_wires: List of wires to measure at the end of the circuit.
    """

    n_features = n_qubits * feats_per_qubit

    if trainable_layers is None:
        trainable_layers = [0] * (reuploads - 1) + [2]
    if isinstance(trainable_layers, int):
        if entangle_between_reuploads:
           trainable_layers = [trainable_layers] * reuploads
        else:
           trainable_layers = [0] * (reuploads - 1) + [trainable_layers]
    else:
        if entangle_between_reuploads and len(trainable_layers) != reuploads:
            raise ValueError(
                f"`trainable_layers` must be either an int or a list of length equal to `reuploads`. "
                f"Got trainable_layers={trainable_layers} and reuploads={reuploads}."
            )
        elif not entangle_between_reuploads:
            raise ValueError(
                f"When `entangle_between_reuploads` is False, `trainable_layers` must be an int. "
                f"Got trainable_layers={trainable_layers}."
            )

    if sum(trainable_layers) == 0:
        raise ValueError("At least one trainable layer is required for the circuit to be trainable.")
    
    if isinstance(ranges, int):
        ranges = [ranges] * reuploads
    elif len(ranges) != reuploads:
        raise ValueError(
            f"`ranges` must be either an int or a list of length equal to `reuploads`. "
            f"Got ranges={ranges} and reuploads={reuploads}."
        )

    if len(ranges) != reuploads:
        raise ValueError(
            f"`ranges` must have length equal to `reuploads`. "
            f"Got len(ranges)={len(ranges)} and reuploads={reuploads}."
        )
    
    if rot_gates is None:
        rot_gates = [qml.RX, qml.RZ]
    
    if measurement_wires is None:
        measurement_wires = [0]

    def feature_block(inputs):
        for q in range(n_qubits):
            for f in range(feats_per_qubit):
                feature_idx = q * feats_per_qubit + f
                gate = rot_gates[f % len(rot_gates)]
                gate(inputs[..., feature_idx], wires=q, id=f"f{feature_idx}")

    def strongly_entangling_block(weights, trainable_layers, ranges, start_idx):
        repeats = max(trainable_layers, 1)

        if isinstance(ranges, int):
            ranges = [ranges] * repeats
        elif len(ranges) != repeats:
            raise ValueError(
                f"`ranges` must be either an int or a list of length equal to `trainable_layers`. "
                f"Got ranges={ranges} and trainable_layers={trainable_layers}."
            )
        
        for layer in range(repeats):
            for q in range(n_qubits):
                if trainable_layers > 0:
                    # Apply trainable Rot gates for this layer
                    for g, gate in enumerate(rot_gates):
                        gate(weights[start_idx + layer, q, g], wires=q, id=f"l{start_idx + layer}g{g}")

            if n_qubits > 1:
                for q in range(n_qubits - 1):
                    if ranges[layer] == 0 or ranges[layer] >= n_qubits:
                        raise ValueError(
                            f"Invalid range value: {ranges[layer]}. "
                            f"Range must be between 1 and {n_qubits - 1} (n_qubits-1)."
                        )

                    # Apply entangling pattern for this layer
                    control = q
                    target = (q + ranges[layer]) % n_qubits
                    ent_gate(wires=[control, target])

                if n_qubits > 2:
                    ent_gate(wires=[n_qubits - 1, 0])

    @qml.qnode(device, interface="torch", diff_method="best")
    def circuit(inputs, weights):
        layer_idx = 0
        for r in range(reuploads):
            feature_block(inputs)
            if r == reuploads - 1:
                qml.Barrier(wires=range(n_qubits), only_visual=True)
            if entangle_between_reuploads or r == reuploads - 1:
                strongly_entangling_block(weights, trainable_layers[r], ranges[r], layer_idx)
            layer_idx += trainable_layers[r]

        return qml.probs(wires=measurement_wires)

    weight_shapes = {
        "weights": (sum(trainable_layers), n_qubits, len(rot_gates))
    }

    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    return qlayer, circuit, weight_shapes, n_features

class BinaryVQC(nn.Module):
    def __init__(self, qml_device: qml.Device, template: int | str = None):
        super().__init__()
        self.template = template
        self.cuda_device = "cpu"
        self.qml_device = qml_device

        self.initialise()

    def initialise(self):
        match(self.template):
            case 1:
                self.qlayer, self.circuit, self.weight_shapes, self.n_features = reuploading_qlayer(
                    n_qubits=2,
                    feats_per_qubit=3,
                    device=self.qml_device,
                    reuploads=3,
                    entangle_between_reuploads=True,
                    trainable_layers=[1, 2, 3]
                )
            case 2:
                self.qlayer, self.circuit, self.weight_shapes, self.n_features = reuploading_qlayer(
                    n_qubits=2,
                    feats_per_qubit=3,
                    device=self.qml_device,
                    reuploads=3,
                    entangle_between_reuploads=False,
                    trainable_layers=[3]
                )
            case 3:
                self.qlayer, self.circuit, self.weight_shapes, self.n_features = reuploading_qlayer(
                    n_qubits=3,
                    feats_per_qubit=2,
                    device=self.qml_device,
                    reuploads=3,
                    entangle_between_reuploads=True,
                    trainable_layers=[1, 2, 3],
                    ranges=1
                )
            case 'deep':
                self.qlayer, self.circuit, self.weight_shapes, self.n_features = reuploading_qlayer(
                    n_qubits=2,
                    feats_per_qubit=2,
                    device=self.qml_device,
                    reuploads=5,
                    entangle_between_reuploads=True,
                    trainable_layers=[1, 2, 2, 3, 3]
                )
            case 'wide':
                self.qlayer, self.circuit, self.weight_shapes, self.n_features = reuploading_qlayer(
                    n_qubits=4,
                    feats_per_qubit=1,
                    device=self.qml_device,
                    reuploads=2,
                    entangle_between_reuploads=True,
                    trainable_layers=[1, 2]
                )
            case 'dense':
                self.qlayer, self.circuit, self.weight_shapes, self.n_features = reuploading_qlayer(
                    n_qubits=2,
                    feats_per_qubit=4,
                    device=self.qml_device,
                    reuploads=2,
                    entangle_between_reuploads=True,
                    trainable_layers=[1, 2]
                )
            case _:
                self.qlayer, self.circuit, self.weight_shapes, self.n_features = reuploading_qlayer(
                    n_qubits=2,
                    feats_per_qubit=2,
                    device=self.qml_device,
                    reuploads=2,
                    entangle_between_reuploads=True,
                    trainable_layers=[1, 2]
                )

    def to(self, device):
        self.cuda_device = device
        self.qlayer.to(device)
        return super().to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = self.qlayer(x)

        half = probs.shape[1] // 2
        return torch.stack([probs[:, :half].sum(dim=1), probs[:, half:].sum(dim=1)], dim=1)[:, 1]
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader = None,
        epochs: int = 100,
        lr: float = 0.01,
        optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
        criterion: nn.Module = nn.BCELoss(),
        patience: int = 10,
        warmup: int = 15,
        min_delta: float = 1e-3,
        plot: bool = True,
        title_prefix: str = "",
        restore_best: bool = True,
        verbose: bool = True,
    ):
        self.to(self.cuda_device)

        optimizer = optimizer_cls(self.parameters(), lr=lr)

        best_state = None
        patience_counter = 0

        train_losses = []
        val_losses = []
        best_loss = float("inf")

        time_start = time.time()
        for epoch in range(epochs):
            # Training
            epoch_loss = 0.0
            self.train()
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                if verbose:
                    print(f"[Epoch {epoch + 1}/{epochs}] Batch {batch_idx + 1}/{len(train_loader)}", end="\r")

                X_batch = X_batch.to(self.cuda_device)
                y_batch = y_batch.to(self.cuda_device)

                optimizer.zero_grad()

                probs = self(X_batch)
                loss = criterion(probs, y_batch.float())

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            if test_loader is not None:
                val_loss = 0.0
                correct = 0
                total = 0
                self.eval()
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.to(self.cuda_device)
                        y_batch = y_batch.to(self.cuda_device)

                        probs = self(X_batch)
                        loss = criterion(probs, y_batch.float())

                        val_loss += loss.item()

                        preds = (probs > 0.5).long()
                        correct += (preds == y_batch).sum().item()
                        total += y_batch.size(0)

                avg_val_loss = val_loss / len(test_loader)
                val_acc = correct / total

                val_losses.append(avg_val_loss)

            # Plot
            if plot:
                clear_output(wait=True)
                plt.figure(figsize=(10, 6))
                plt.plot(train_losses, label="Train Loss")
                if test_loader is not None:
                    plt.plot(val_losses, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{title_prefix}Training {'and Validation' if test_loader is not None else ''} Loss")
                plt.legend()
                plt.grid(True)
                plt.xlim(0, max(1, len(train_losses) - 1))
                plt.show()

            if verbose:
                print(f"[Epoch {epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f} {f'| Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f}' if test_loader is not None else ''}")
            
            # Early stopping
            monitor_loss = avg_val_loss if test_loader is not None else avg_train_loss
            if epoch >= warmup:
                if monitor_loss < best_loss - min_delta:
                    best_loss = monitor_loss
                    patience_counter = 0

                    if restore_best:
                        best_state = copy.deepcopy(self.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if monitor_loss < best_loss:
                    best_loss = monitor_loss

                    if restore_best:
                        best_state = copy.deepcopy(self.state_dict())

        if restore_best and best_state is not None:
            self.load_state_dict(best_state)

        self.training_time = TimeInt(time.time() - time_start)
        if verbose:
            print(f"Training time: {self.training_time}")

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.best_loss = best_loss

    @property
    def weight(self):
        if hasattr(self, "val_report"):
            w = self.val_report['macro avg']['f1-score']
            return min(max(w, 0.01), 1.0)
        return 1.0
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(X.to(self.cuda_device)).cpu()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return (self.predict_proba(X) > 0.5).long()

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        preds = self.predict(X)
        return accuracy_score(y, preds)

class QuantumECOC:
    """
    Quantum Error-Correcting Output Codes ensemble classifier for multi-class classification using binary variational quantum classifiers as base learners.
    """
    def __init__(self, n_learners: int = None, templates: int | str | list[int | str] = None, device: qml.Device = None, scaler_range: tuple[float, float] = (0, np.pi)):
        self.n_learners = len(templates) if isinstance(templates, list) else n_learners
        self.templates = templates

        self.cuda_device = "cpu"
        self.qml_device = device if device else qml.device("default.qubit", wires=2)
        self.classifiers = []

        self.scaler = MinMaxScaler(feature_range=scaler_range)

    def to(self, device):
        self.cuda_device = device
        for clf in self.classifiers:
            clf.to(device)
        return self
    
    def initialise_ensemble(self, X: DataFrame, y: Series, verbose: bool):
        self.features = X.columns.tolist()
        self.features_to_int = {feat: idx for idx, feat in enumerate(self.features)}
        self.int_to_features = {idx: feat for idx, feat in enumerate(self.features)}
        if verbose:
            print(f"Feature mapping: {self.features_to_int}")

        self.labels = sorted(y.unique())
        self.label_to_int = {label: idx for idx, label in enumerate(self.labels)}
        self.int_to_label = {idx: label for idx, label in enumerate(self.labels)}
        if verbose:
            print(f"Label mapping: {self.label_to_int}")

        if self.n_learners is None:
            self.n_learners = 2 * len(self.labels)

        self.ecoc = build_ecoc_matrix(len(self.labels), self.n_learners)
        self.feat_map = None

    def initialise_classifiers(self):
        if self.templates is not None:
            if isinstance(self.templates, int) or isinstance(self.templates, str):
                self.templates = [self.templates] * self.n_learners
            else:
                if len(self.templates) != self.n_learners:
                    raise ValueError(f"Length of templates does not match n_learners. Got {len(self.templates)} templates and n_learners={self.n_learners}.")
            self.classifiers = [BinaryVQC(self.qml_device, learner).to(self.cuda_device) for learner in self.templates]
        else:
            self.classifiers = [BinaryVQC(self.qml_device).to(self.cuda_device) for _ in range(self.n_learners)]
        
        # pick a list of random features for each classifier
        if self.feat_map is None:
            feats = [i for i in range(len(self.features))]
            total_feats_needed = sum(clf.n_features for clf in self.classifiers)
            final_feats = feats.copy()
            while len(final_feats) < total_feats_needed:
                np.random.shuffle(feats)
                final_feats += feats
            self.feat_map = [[final_feats.pop() for _ in range(clf.n_features)] for clf in self.classifiers]

        for i, clf in enumerate(self.classifiers):
            clf.feats = self.feat_map[i]
            clf.code = self.ecoc[i]

    def train_ensemble(self, X: np.ndarray, y: Series, X_test: np.ndarray, y_test: Series, bagging: tuple[float, int, int] | None, plot: bool, verbose: bool, fold: int = 0, **fit_params):
        for i, clf in enumerate(self.classifiers):
            if verbose:
                print(f"\nTraining classifier {i + 1}/{self.n_learners}")

            y_train = y.apply(lambda x: clf.code[x])

            if bagging is None:
                k = 1.0
            else:
                k = min(min(max(len(X)*bagging[0], bagging[1]), bagging[2]), len(X)) / len(X)
            if verbose:
                print(f"Using {k*len(X)}//{len(X)} samples")
            if k >= 1.0:
                samples = np.arange(len(X))
            else:
                samples, _ = train_test_split(range(len(X)), train_size=k, stratify=y_train, random_state=2+i+(1000*fold))

            X_tr = torch.tensor(np.array([X[samples, feat] for feat in clf.feats]).T, dtype=torch.float32).to(self.cuda_device)
            y_tr = torch.tensor(y_train.iloc[samples].values, dtype=torch.long).to(self.cuda_device)
            trainloader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

            if X_test is not None and y_test is not None:
                X_te = torch.tensor(np.array([X_test[:, feat] for feat in clf.feats]).T, dtype=torch.float32).to(self.cuda_device)
                y_te = torch.tensor(y_test.apply(lambda x: clf.code[x]).values, dtype=torch.long).to(self.cuda_device)
                testloader = DataLoader(TensorDataset(X_te, y_te), batch_size=64, shuffle=False)
            else:
                testloader = None

            clf.fit(trainloader, testloader, title_prefix=f"Classifier {i + 1}/{self.n_learners}: ", plot=plot, verbose=verbose, **fit_params)

            if testloader is not None:
                clf.val_probs = clf.predict_proba(X_te)
                clf.val_report = ValReport(classification_report(y_test.apply(lambda x: clf.code[x]), (clf.val_probs > 0.5).long(), zero_division=0, output_dict=True))

        if plot:
            clear_output(wait=True)
            self.plot()

    def plot(self):
        plt.figure(figsize=(10, 6))
        for i, clf in enumerate(self.classifiers):
            plt.plot(clf.val_losses, label=f"Classifier {i + 1} (Code: {clf.code})")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss for Each Classifier")
        plt.xlim(0, max(1, max(len(clf.val_losses) for clf in self.classifiers) - 1))
        plt.legend()
        plt.grid(True)
        plt.show()

    def fit(self, X: DataFrame, y: Series, X_test: DataFrame = None, y_test: Series = None, bagging: tuple[float, int, int] | None = (0.5, 512, 2048), plot: bool = False, verbose: bool = False, **fit_params):
        start_time = time.time()
        self.initialise_ensemble(X, y, verbose)
        self.initialise_classifiers()

        X = self.scaler.fit_transform(X)
        if X_test is not None:
            X_test = self.scaler.transform(X_test)

        y = y.map(self.label_to_int)
        if y_test is not None:
            y_test = y_test.map(self.label_to_int)

        self.train_ensemble(X, y, X_test, y_test, bagging, plot, verbose, **fit_params)

        self.training_time = TimeInt(time.time() - start_time)
        if verbose:
            print(f"\nTotal training time after fitting {self.n_learners} classifiers: {self.training_time}")

    @property
    def learner_weights(self):
        if all(hasattr(clf, "weight") for clf in self.classifiers):
            return [clf.weight for clf in self.classifiers]
        return None
    
    def predict(self, X: DataFrame) -> np.ndarray:
        X = self.scaler.transform(X)
        with torch.no_grad():
            final_preds = np.array([[0.0] * len(self.labels)] * len(X))
            for clf in self.classifiers:
                X_tr = torch.tensor(np.array([X[:, feat] for feat in clf.feats]).T, dtype=torch.float32).to(self.cuda_device)

                preds = clf.predict(X_tr)
                for j, pred in enumerate(preds):
                    final_preds[j][clf.code == pred.item()] += clf.weight

            return [self.int_to_label[pred.item()] for pred in np.array(final_preds).argmax(axis=1)]
    
    def predict_proba(self, X: DataFrame) -> np.ndarray:
        X = self.scaler.transform(X)
        with torch.no_grad():
            final_preds = np.array([[0.0] * len(self.labels)] * len(X))
            for clf in self.classifiers:
                X_tr = torch.tensor(np.array([X[:, feat] for feat in clf.feats]).T, dtype=torch.float32).to(self.cuda_device)

                p1 = clf.predict_proba(X_tr).numpy()
                p0 = 1.0 - p1

                final_preds[:, clf.code == 0] += p0[:, None] * clf.weight
                final_preds[:, clf.code == 1] += p1[:, None] * clf.weight

            return final_preds / sum(clf.weight for clf in self.classifiers)
    
    def score(self, X: DataFrame, y: Series) -> float:
        preds = self.predict(X)
        return accuracy_score(y.values, preds)

class StackedECOC(QuantumECOC):
    """
    Stacked Quantum Ensemble that extends QuantumECOC by adding a meta-learner on top of the quantum base learners.
    """
    def __init__(self, meta_learner = None, n_learners: int = None, templates: list[int | str] = None, device: qml.Device = None, **kwargs):
        super().__init__(n_learners, templates, device, **kwargs)
        self.meta_learner = meta_learner if meta_learner is not None else SVC(kernel="rbf", random_state=2)

    def fit(self, X: DataFrame, y: Series, X_test: DataFrame = None, y_test: Series = None, k_folds: int = 5, plot: bool = False, verbose: bool = False, **fit_params):
        start_time = time.time()
        self.initialise_ensemble(X, y, verbose=verbose)

        class_samples = y.value_counts()
        k_folds = min(k_folds, class_samples.min())
        if k_folds > 1:
            X_meta = np.zeros((len(X), self.n_learners))
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=2)
            for f, (train_index, val_index) in enumerate(skf.split(X, y)):
                if verbose:
                    print(f"\nFold {f + 1}/{k_folds}")
                
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                self.initialise_classifiers()

                fold_scaler = MinMaxScaler(feature_range=self.scaler.feature_range)
                X_train = fold_scaler.fit_transform(X_train)
                X_val = fold_scaler.transform(X_val)

                y_train = y_train.map(self.label_to_int)
                y_val = y_val.map(self.label_to_int)

                self.train_ensemble(X_train, y_train, X_val, y_val, plot=False, verbose=False, fold=f, **fit_params)

                X_fold = np.array([
                    clf.val_probs.numpy() # maybe * clf.weight later
                    for clf in self.classifiers
                ]).T

                X_meta[val_index, :] = X_fold
            self.meta_learner.fit(X_meta, y.map(self.label_to_int))

            self.initialise_classifiers()

            X = self.scaler.fit_transform(X)
            if X_test is not None:
                X_test = self.scaler.transform(X_test)

            y = y.map(self.label_to_int)
            if y_test is not None:
                y_test = y_test.map(self.label_to_int)

            self.train_ensemble(X, y, X_test, y_test, plot, verbose, **fit_params)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
            
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)

            y_train = y_train.map(self.label_to_int)
            y_val = y_val.map(self.label_to_int)

            self.initialise_classifiers()
            self.train_ensemble(X_train, y_train, X_val, y_val, plot, verbose, **fit_params)

            X_meta = np.array([clf.val_probs for clf in self.classifiers]).T
            self.meta_learner.fit(X_meta, y_val)

        self.training_time = time.time() - start_time
        if verbose:
            print(f"\nTotal training time after fitting {self.n_learners} classifiers across {k_folds} folds: {self.training_time:.2f} seconds")

    def predict(self, X: DataFrame) -> np.ndarray:
        X = self.scaler.transform(X)
        meta_X = np.array(
            [
                clf.predict_proba(
                    torch.tensor(
                        np.array([X[:, feat] for feat in clf.feats]).T,
                        dtype=torch.float32
                    ).to(self.cuda_device)
                ) for clf in self.classifiers
            ]
        ).T
        return [self.int_to_label[pred] for pred in self.meta_learner.predict(meta_X)]
    
    def predict_proba(self, X: DataFrame) -> np.ndarray:
        X = self.scaler.transform(X)
        meta_X = np.array(
            [
                clf.predict_proba(
                    torch.tensor(
                        np.array([X[:, feat] for feat in clf.feats]).T,
                        dtype=torch.float32
                    ).to(self.cuda_device)
                ) for clf in self.classifiers
            ]
        ).T
        return self.meta_learner.predict_proba(meta_X)
    
    def score(self, X: DataFrame, y: Series) -> float:
        preds = self.predict(X)
        return accuracy_score(y.values, preds)
    
class MetaVQC:
    """
    Quantum meta-learner for StackedECOC that takes the outputs of the base learners as input and learns to optimally combine their predictions.
    """
    def __init__(self):
        pass

class CoherentECOC(QuantumECOC):
    """
    Measurement Free Quantum Ensemble that extends QuantumECOC by combining the outputs of the base learners without collapsing their quantum states.
    """
    def __init__(self, n_learners: int = None, templates: list[int | str] = None, device: qml.Device = None, **kwargs):
        super().__init__(n_learners, templates, device, **kwargs)

if __name__ == "__main__":
    import prince
    from ucimlrepo import fetch_ucirepo
    import pandas as pd

    # iris = fetch_ucirepo(id=53)

    # X = iris.data.features
    # y = iris.data.targets['class']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

    # ens = CsQuECOC().to("cpu")
    # ens.fit(X_train, y_train, epochs=200, plot=False, verbose=True)

    # train_score = ens.score(X_train, y_train)
    # print(f"Training Accuracy: {train_score:.2f}")
    # print("Testing Classification Report:\n", classification_report(y_test, ens.predict(X_test), zero_division=0))

    # ---

    # lab_train = pd.read_csv('lab1-train-processed.csv')
    # lab_test = pd.read_csv('lab1-test-processed.csv')

    # y_train = lab_train['class']
    # y_test = lab_test['class']
    # X_train = lab_train.drop('class', axis=1)
    # X_test = lab_test.drop('class', axis=1)

    # top_cols = [col for col in X_train.columns if col.startswith('X')]
    # famd_cols = [col for col in X_train.columns if col.startswith('C')]

    # ens = CsQuECOC(feats_per_qubit=3, reuploads=3, trainable_layers=[1,2,3]).to("cpu")
    # ens.fit(X_train, y_train, epochs=200, plot=False, verbose=True)

    # train_score = ens.score(X_train, y_train)
    # print(f"Training Accuracy: {train_score:.2f}")
    # print("Testing Classification Report:\n", classification_report(y_test, ens.predict(X_test), zero_division=0))

    X = pd.read_csv(f"datasets/0/iris.csv")
    y = X['target']
    X = X.drop('target', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

    # ens = QuantumECOC().to("cpu")
    # ens.fit(X_train, y_train, X_test, y_test, epochs=200, plot=False, verbose=True)
    # print(f"QuantumECOC Classification Report:\n{classification_report(y_test, ens.predict(X_test), zero_division=0)}\n")

    ens = StackedECOC().to("cpu")
    ens.fit(X_train, y_train, X_test, y_test, epochs=200, plot=False, verbose=True)
    print(f"StackedECOC Classification Report:\n{classification_report(y_test, ens.predict(X_test), zero_division=0)}\n")

    plt.figure(figsize=(10, 6))
    for j, clf in enumerate(ens.classifiers):
        plt.plot(clf.val_losses, label=f"Classifier {j + 1} (Code: {ens.ecoc[j]})")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss for Each Classifier")
    plt.xlim(0, max(1, max(len(clf.val_losses) for clf in ens.classifiers) - 1))
    plt.legend()
    plt.grid(True)
    plt.show()