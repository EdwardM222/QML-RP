import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from pennylane import numpy as np
from scipy.linalg import hadamard
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pandas import DataFrame, Series
import  warnings
warnings.filterwarnings(
    "ignore",
    message=r"The 'id' argument is deprecated and will be removed in v0\.46\.",
    category=qml.exceptions.PennyLaneDeprecationWarning,
    module=r"pennylane\.operation",
)

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
    def __init__(self, qlayer: qml.qnn.TorchLayer):
        super().__init__()
        self.qlayer = qlayer
        self.device = "cpu"

    def to(self, device):
        self.device = device
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
        self.to(self.device)

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

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                probs = self(X_batch)
                loss = criterion(probs, y_batch.float())

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            if test_loader:
                val_loss = 0.0
                correct = 0
                total = 0
                self.eval()
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

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
                if test_loader:
                    plt.plot(val_losses, label="Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{title_prefix}Training {'and Validation' if test_loader else ''} Loss")
                plt.legend()
                plt.grid(True)
                plt.xlim(0, max(1, len(train_losses) - 1))
                plt.show()

            if verbose:
                print(f"[Epoch {epoch + 1}/{epochs}] Train Loss: {avg_train_loss:.4f} {f'| Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f}' if test_loader else ''}")
            
            # Early stopping
            monitor_loss = avg_val_loss if test_loader else avg_train_loss
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

        self.training_time = time.time() - time_start
        if verbose:
            print(f"Training time: {self.training_time:.2f} seconds")

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.best_loss = best_loss
    
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(X.to(self.device)).cpu()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return (self.predict_proba(X) > 0.5).long()

    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        preds = self.predict(X)
        return accuracy_score(y, preds)

class QuECOC:
    """
    Quantum Error-Correcting Output Codes ensemble classifier for multi-class classification using binary variational quantum classifiers as base learners.
    """
    def __init__(self, n_learners: int = None, device: qml.Device = None, **kwargs):
        self.n_learners = n_learners

        self.cuda_device = "cpu"
        self.qml_device = device if device else qml.device("default.qubit", wires=2)
        self.classifiers = []

        self.kwargs = kwargs

    def to(self, device):
        self.cuda_device = device
        for clf in self.classifiers:
            clf.to(device)
        return self
    
    def initialise(self):
        self.ecoc = build_ecoc_matrix(len(self.labels), self.n_learners)
        for i in range(self.n_learners):
            qlayer, circuit, weight_shapes, n_features = reuploading_qlayer(
                n_qubits=self.kwargs.get("n_qubits", 2),
                feats_per_qubit=self.kwargs.get("feats_per_qubit", 2),
                device=self.qml_device,
                reuploads=self.kwargs.get("reuploads", 2),
                entangle_between_reuploads=self.kwargs.get("entangle_between_reuploads", True),
                trainable_layers=self.kwargs.get("trainable_layers", 2),
                ranges=self.kwargs.get("ranges", 1),
                rot_gates=self.kwargs.get("rot_gates", [qml.RX, qml.RZ]),
                ent_gate=self.kwargs.get("ent_gate", qml.CZ),
                measurement_wires=self.kwargs.get("measurement_wires", [0])
            )
            clf = BinaryVQC(qlayer).to(self.cuda_device)
            clf.n_qubits = self.kwargs.get("n_qubits", 2)
            clf.weight_shapes = weight_shapes
            clf.n_features = n_features
            self.classifiers.append(clf)

    def fit(self, X: DataFrame, y: Series, X_test: DataFrame = None, y_test: Series = None, scaler_range: tuple = (0, np.pi), plot: bool = False, verbose: bool = False, **fit_params):
        self.features = X.columns.tolist()
        self.features_to_int = {feat: idx for idx, feat in enumerate(self.features)}
        self.int_to_features = {idx: feat for idx, feat in enumerate(self.features)}
        if verbose:
            print(f"Feature mapping: {self.features_to_int}")
        
        self.scaler = MinMaxScaler(feature_range=scaler_range)
        X = self.scaler.fit_transform(X)

        if X_test is not None:
            X_test = self.scaler.transform(X_test)
        
        self.labels = sorted(y.unique())
        self.label_to_int = {label: idx for idx, label in enumerate(self.labels)}
        self.int_to_label = {idx: label for idx, label in enumerate(self.labels)}
        if verbose:
            print(f"Label mapping: {self.label_to_int}")
        y = y.map(self.label_to_int)

        if y_test is not None:
            y_test = y_test.map(self.label_to_int)

        if self.n_learners is None:
            self.n_learners = 2 * len(self.labels)
        self.initialise()

        # pick a list of random features for each classifier
        feats = [i for i in range(X.shape[1])]
        total_feats_needed = self.kwargs.get("n_qubits", 2) * self.kwargs.get("feats_per_qubit", 2) * self.n_learners
        final_feats = feats.copy()
        while len(final_feats) < total_feats_needed:
            np.random.shuffle(feats)
            final_feats += feats

        for i, clf in enumerate(self.classifiers):
            if verbose:
                print(f"\nTraining classifier {i + 1}/{self.n_learners}")

            clf.feats = [final_feats.pop() for _ in range(clf.n_features)]
            X_tr = torch.tensor(np.array([X[:, feat] for feat in clf.feats]).T, dtype=torch.float32).to(self.cuda_device)

            if X_test is not None:
                X_te = torch.tensor(np.array([X_test[:, feat] for feat in clf.feats]).T, dtype=torch.float32).to(self.cuda_device)

            code = self.ecoc[i]
            y_train = y.apply(lambda x: code[x])
            y_tr = torch.tensor(y_train.values, dtype=torch.long).to(self.cuda_device)

            if X_test is not None and y_test is not None:
                y_te = torch.tensor(y_test.apply(lambda x: code[x]).values, dtype=torch.long).to(self.cuda_device)
                testloader = DataLoader(TensorDataset(X_te, y_te), batch_size=32, shuffle=False)
            else:
                testloader = None

            trainloader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

            clf.fit(trainloader, testloader, title_prefix=f"Classifier {i + 1}/{self.n_learners}: ", plot=plot, verbose=verbose, **fit_params)

            if testloader:
                clf.val_probs = clf.predict_proba(X_te)
                clf.val_report = classification_report(y_test.apply(lambda x: code[x]), (clf.val_probs > 0.5).long(), zero_division=0, output_dict=True)

        if plot:
            clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            for j, clf in enumerate(self.classifiers):
                plt.plot(clf.val_losses, label=f"Classifier {j + 1} (Code: {self.ecoc[j]})")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title("Validation Loss for Each Classifier")
            plt.xlim(0, max(1, max(len(clf.val_losses) for clf in self.classifiers) - 1))
            plt.legend()
            plt.grid(True)
            plt.show()

        if verbose:
            print(f"\nTotal training time after fitting {self.n_learners} classifiers: {sum(clf.training_time for clf in self.classifiers):.2f} seconds")

    def predict(self, X: DataFrame) -> np.ndarray:
        X = self.scaler.transform(X)
        with torch.no_grad():
            final_preds = np.array([[0] * len(self.labels)] * len(X))
            for i, clf in enumerate(self.classifiers):
                X_tr = torch.tensor(np.array([X[:, feat] for feat in clf.feats]).T, dtype=torch.float32).to(self.cuda_device)

                preds = clf.predict(X_tr)
                for j, pred in enumerate(preds):
                    final_preds[j][self.ecoc[i] == pred.item()] += 1

            return [self.int_to_label[pred.item()] for pred in np.array(final_preds).argmax(axis=1)]
    
    def predict_proba(self, X: DataFrame) -> np.ndarray:
        X = self.scaler.transform(X)
        with torch.no_grad():
            final_preds = np.array([[0.0] * len(self.labels)] * len(X))
            for i, clf in enumerate(self.classifiers):
                X_tr = torch.tensor(np.array([X[:, feat] for feat in clf.feats]).T, dtype=torch.float32).to(self.cuda_device)

                p1 = clf.predict_proba(X_tr).numpy()
                p0 = 1.0 - p1

                final_preds[:, self.ecoc[i] == 0] += p0[:, None]
                final_preds[:, self.ecoc[i] == 1] += p1[:, None]

            return np.array(final_preds) / self.n_learners
    
    def score(self, X: DataFrame, y: Series) -> float:
        preds = self.predict(X)
        return accuracy_score(y.values, preds)

class CsQuECOC(QuECOC):
    """
    Classical Stacked Quantum Error-Correcting Output Codes ensemble that extends QuECOC by adding a classical meta-learner on top of the quantum base learners.
    """
    def __init__(self, meta_learner = None, n_learners: int = None, device: qml.Device = None, **kwargs):
        super().__init__(n_learners, device, **kwargs)
        self.meta_learner = meta_learner if meta_learner is not None else SVC(kernel="rbf", class_weight='balanced', random_state=2)

    def fit(self, X: DataFrame, y: Series, plot: bool = False, verbose: bool = False, **fit_params):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
        super().fit(X_train, y_train, X_val, y_val, plot=plot, verbose=verbose, **fit_params)

        meta_X_train = np.array([clf.val_probs for clf in self.classifiers]).T
        self.meta_learner.fit(meta_X_train, y_val.map(self.label_to_int))

    def predict(self, X: DataFrame) -> np.ndarray:
        X = self.scaler.transform(X)
        meta_X = np.array([clf.predict_proba(torch.tensor(np.array([X[:, feat] for feat in clf.feats]).T, dtype=torch.float32).to(self.cuda_device)) for clf in self.classifiers]).T
        return [self.int_to_label[pred] for pred in self.meta_learner.predict(meta_X)]
    
    def predict_proba(self, X: DataFrame) -> np.ndarray:
        X = self.scaler.transform(X)
        meta_X = np.array([clf.predict_proba(torch.tensor(np.array([X[:, feat] for feat in clf.feats]).T, dtype=torch.float32).to(self.cuda_device)) for clf in self.classifiers]).T
        return self.meta_learner.predict_proba(meta_X)
    
    def score(self, X: DataFrame, y: Series) -> float:
        preds = self.predict(X)
        return accuracy_score(y.values, preds)

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

    CQens = CsQuECOC(feats_per_qubit=3, reuploads=3, trainable_layers=[1,2,3]).to("cpu")
    CQens.fit(X_train, y_train, epochs=200, plot=False, verbose=True)
    print(f"CsQuECOC Classification Report:\n{classification_report(y_test, CQens.predict(X_test), zero_division=0)}\n")