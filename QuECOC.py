import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pandas import DataFrame, Series
from itertools import product
from random import sample, choices
import json
from pathlib import Path
from typeguard import typechecked

from Ansatze import AnsatzSpec

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

@typechecked
def build_ecoc_matrix(n_classes: int, n_learners: int, depth: int = 2) -> np.ndarray:
    if depth < 2:
        raise ValueError("depth must be at least 2")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if n_learners < 1:
        raise ValueError("n_learners must be at least 1")
    if depth > n_classes:
        raise ValueError("depth cannot be greater than n_classes")
    
    m = int(np.ceil(np.log(n_classes) / np.log(depth)))

    X = np.array(list(product(range(depth), repeat=m)), dtype=int)[:n_classes]

    A_all = np.array(list(product(range(depth), repeat=m)), dtype=int)
    A_all = A_all[np.any(A_all != 0, axis=1)]
    A = A_all[:n_learners]

    ecoc = (A @ X.T) % depth

    temp = ecoc.copy()
    while n_learners > len(ecoc):
        temp = np.roll(temp, 1, axis=1)
        ecoc = np.append(ecoc, temp, axis=0)

    ecoc = ecoc[:n_learners, :]
    for code in ecoc:
        mapping = {old: new for new, old in enumerate(sorted(set(code)))}
        for i in range(len(code)):
            code[i] = mapping[code[i]]

    return ecoc

@typechecked
class VQC(nn.Module):
    def __init__(
            self,
            n_classes: int = 2,
            feats: list[int] | None = None,
            qml_device: str | qml.devices.Device = "default.qubit",
            template: int | str = 0,
            measurement_mode: str = "min",
            **kwargs
        ):
        super().__init__()
        self.n_classes = n_classes
        self.feats = feats
        self.qml_device = qml_device
        self.template = template
        self.measurement_mode = measurement_mode

        self.cuda_device = "cpu"

        self.kwargs = kwargs

        self.initialise()

    def to(self, device):
        self.cuda_device = device
        if hasattr(self, "qlayer") and self.qlayer is not None:
            self.qlayer.to(device)
        return super().to(device)
    
    def initialise(self):
        if self.template == "manual":
            self.qlayer = self.kwargs.get("qlayer", None)
            self.circuit = self.kwargs.get("circuit", None)
            self.weight_shapes = self.kwargs.get("weight_shapes", None)
            self.n_features = self.kwargs.get("n_features", None)
            self.n_qubits = self.kwargs.get("n_qubits", 2)
            self.ansatz_spec = self.kwargs.get("ansatz_spec", None)
            return

        n_qubits = self.kwargs.get("n_qubits", 2)

        self.ansatz_spec = AnsatzSpec.from_template(
            template=self.template,
            n_qubits=n_qubits,
            input_dim=len(self.feats)
        )

        self.weight_shapes = self.ansatz_spec.weight_shapes
        self.n_features = self.ansatz_spec.n_features
        self.n_qubits = self.ansatz_spec.n_qubits

        self.train_losses = []
        self.val_losses = []
        self.best_loss = float("inf")
        self.optimizer = None

    def materialise(self):
        if hasattr(self, "qlayer") and self.qlayer is not None:
            return

        if not hasattr(self, "ansatz_spec") or self.ansatz_spec is None:
            raise RuntimeError("Cannot materialise VQC because ansatz_spec is None.")

        self.qlayer, self.circuit, self.weight_shapes, self.n_features = (
            self.ansatz_spec.build_qlayer(
                device_name=self.qml_device,
                measurement_wires=self.n_qubits if self.measurement_mode == "full" else int(np.ceil(np.log2(self.n_classes)))
            )
        )

        self.qlayer.to(self.cuda_device)

    def dematerialise(self):
        self.qlayer = None
        self.circuit = None
        self.optimizer = None

    def reset_model(self):
        self.dematerialise()

        self.train_losses = []
        self.val_losses = []
        self.best_loss = float("inf")
        self.optimizer = None

        for attr in [
            "val_probs",
            "val_report",
            "training_time",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.materialise()
        meas = self.qlayer(x)

        states_per_class = meas.shape[1] // self.n_classes
        probs = torch.zeros((meas.shape[0], self.n_classes), device=meas.device)
        for i in range(self.n_classes):
            p = meas[:, i*states_per_class:(i+1)*states_per_class].sum(dim=1)
            probs[:, i] = p
        return probs

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        epochs: int = 100,
        lr: float = 0.01,
        optimizer_cls: type = torch.optim.Adam,
        criterion: nn.Module = nn.NLLLoss(),
        patience: int = 10,
        warmup: int = 15,
        min_delta: float = 1e-3,
        plot: bool = False,
        title_prefix: str = "",
        restore_best: bool = True,
        verbose: bool = False,
    ):
        X_tr = torch.tensor(X, dtype=torch.float32).to(self.cuda_device)
        y_tr = torch.tensor(y, dtype=torch.long).to(self.cuda_device)
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

        if X_test is not None and y_test is not None:
            X_te = torch.tensor(X_test, dtype=torch.float32).to(self.cuda_device)
            y_te = torch.tensor(y_test, dtype=torch.long).to(self.cuda_device)
            test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=64, shuffle=False)
        else:
            test_loader = None
        
        self.materialise()

        if self.optimizer is None:
            self.optimizer = optimizer_cls(self.parameters(), lr=lr)

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

                self.optimizer.zero_grad()

                probs = self(X_batch)
                loss = criterion(torch.log(probs.clamp(min=1e-8)), y_batch)

                loss.backward()
                self.optimizer.step()

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
                        loss = criterion(torch.log(probs.clamp(min=1e-8)), y_batch)

                        val_loss += loss.item()

                        preds = probs.argmax(dim=1)
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

        self.train_losses.extend(train_losses)
        self.val_losses.extend(val_losses)
        self.best_loss = min(best_loss, self.best_loss)
        
        if X_test is not None and y_test is not None:
            self.val_probs = self.predict_proba(X_test)
            self.val_report = ValReport(classification_report(y_test, self.val_probs.argmax(axis=1), zero_division=0, output_dict=True))

        return self.qlayer.weights.detach().cpu().numpy()

    @property
    def weight(self) -> float:
        if hasattr(self, "val_report"):
            w = self.val_report['macro avg']['f1-score']
            return float(min(max(w, 0.01), 1.0))
        return 1.0
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.cuda_device)
            return self(X_t.to(self.cuda_device)).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X).argmax(axis=1))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return accuracy_score(y, preds)

@typechecked
class QuantumECOC:
    """
    Quantum Error-Correcting Output Codes ensemble classifier for multi-class classification using binary variational quantum classifiers as base learners.
    """
    def __init__(
        self,
        n_learners: int | None = None,
        templates: int | str | list[int | str] = "default",
        feature_density: int | float = 1.0,
        ecoc_depth: int = 2,
        device: str = "default.qubit",
        scaler_range: tuple[float, float] = (0, np.pi)
    ):
        self.n_learners = len(templates) if isinstance(templates, list) else n_learners
        self.templates = templates
        self.feature_density = feature_density
        self.ecoc_depth = ecoc_depth

        self.cuda_device = "cpu"
        self.qml_device = device
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
        self.n_classes = len(self.labels)
        self.label_to_int = {label: idx for idx, label in enumerate(self.labels)}
        self.int_to_label = {idx: label for idx, label in enumerate(self.labels)}
        if verbose:
            print(f"Label mapping: {self.label_to_int}")

        if self.n_learners is None:
            self.n_learners = 2 * len(self.labels)

        self.ecoc = build_ecoc_matrix(len(self.labels), self.n_learners, self.ecoc_depth)
        self.feat_map = []

        if isinstance(self.templates, list):
            if len(self.templates) != self.n_learners:
                raise ValueError(f"Length of templates does not match n_learners. Got {len(self.templates)} templates and n_learners={self.n_learners}.")
        else:
            self.templates = [self.templates] * self.n_learners
        
        if isinstance(self.feature_density, float):
            if not (0 < self.feature_density <= 1):
                raise ValueError(f"feature_density must be in the range (0, 1]. Got {self.feature_density}.")
            k = max(1, int(self.feature_density * len(self.features)))
        elif isinstance(self.feature_density, int):
            if not (1 <= self.feature_density <= len(self.features)):
                raise ValueError(f"feature_density must be in the range [1, {len(self.features)}]. Got {self.feature_density}.")
            k = self.feature_density
            
        for i in range(self.n_learners):
            self.classifiers.append(
                VQC(
                    n_classes=len(set(self.ecoc[i])),
                    feats=sample(range(len(self.features)), k=k),
                    qml_device=self.qml_device,
                    template=self.templates[i]
                ).to(self.cuda_device)
            )

    def train_ensemble(
        self,
        X: np.ndarray,
        y: Series,
        X_test: np.ndarray | None = None,
        y_test: Series | None = None,
        bagging: tuple[float, int, int] | None = None, 
        plot: bool = False,
        verbose: bool = False,
        fold: int = 0,
        **fit_params
    ):
        for i, clf in enumerate(self.classifiers):
            if verbose:
                print(f"\nTraining classifier {i + 1}/{self.n_learners}")

            y_train = y.apply(lambda x: self.ecoc[i][x])

            if bagging is None:
                k = 1.0
            else:
                k = min(min(max(len(X)*bagging[0], bagging[1]), bagging[2]), len(X)) / len(X)
            if verbose:
                print(f"Using {int(k*len(X))}/{len(X)} samples")
            if k >= 1.0:
                samples = np.arange(len(X))
            else:
                samples, _ = train_test_split(range(len(X)), train_size=k, stratify=y_train, random_state=2+i+(1000*fold))

            X_tr = np.array([X[samples, feat] for feat in clf.feats]).T
            y_tr = y_train.iloc[samples].values
            X_te = np.array([X_test[:, feat] for feat in clf.feats]).T
            y_te = y_test.apply(lambda x: self.ecoc[i][x]).values

            clf.fit(X_tr, y_tr, X_te, y_te, title_prefix=f"Classifier {i + 1}/{self.n_learners}: ", plot=plot, verbose=verbose, **fit_params)

        if plot:
            clear_output(wait=True)
            self.plot()

    def plot(self):
        plt.figure(figsize=(10, 6))
        for i, clf in enumerate(self.classifiers):
            plt.plot(clf.val_losses, label=f"Classifier {i + 1} (Code: {self.ecoc[i]})")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss for Each Classifier")
        plt.xlim(0, max(1, max(len(clf.val_losses) for clf in self.classifiers) - 1))
        plt.legend()
        plt.grid(True)
        plt.show()

    def fit(
        self,
        X: DataFrame,
        y: Series,
        X_test: DataFrame | None = None,
        y_test: Series | None = None,
        bagging: tuple[float, int, int] | None = (0.5, 512, 2048),
        parallel: bool = False,
        n_jobs: int = -1,
        plot: bool = False,
        verbose: bool = False,
        **fit_params
    ):
        start_time = time.time()
        self.initialise_ensemble(X, y, verbose)

        X_s = self.scaler.fit_transform(X)
        X_test_s = self.scaler.transform(X_test) if X_test is not None else None

        y_m = y.map(self.label_to_int)
        y_test_m = y_test.map(self.label_to_int) if y_test is not None else None

        self.train_ensemble(
            X_s,
            y_m,
            X_test_s,
            y_test_m,
            bagging=bagging,
            plot=plot,
            verbose=verbose,
            **fit_params,
        )

        self.training_time = TimeInt(time.time() - start_time)
        if verbose:
            print(f"\nTotal training time after fitting {self.n_learners} classifiers: {self.training_time}")

    @property
    def learner_weights(self):
        if all(hasattr(clf, "weight") for clf in self.classifiers):
            return [clf.weight for clf in self.classifiers]
        return None
    
    def predict(self, X: DataFrame) -> np.ndarray:
        X_s = self.scaler.transform(X)

        final_preds = np.array([[0.0] * len(self.labels)] * len(X_s))
        for i, clf in enumerate(self.classifiers):
            X_clf = X_s[:, clf.feats]
            preds = clf.predict(X_clf)
            for j, pred in enumerate(preds):
                final_preds[j, self.ecoc[i] == pred.item()] += clf.weight

        final_preds /= sum(clf.weight for clf in self.classifiers)
        return np.array([self.int_to_label[pred] for pred in final_preds.argmax(axis=1)])
    
    def predict_proba(self, X: DataFrame) -> np.ndarray:
        X_s = self.scaler.transform(X)

        final_preds = np.array([[0.0] * len(self.labels)] * len(X_s))
        for i, clf in enumerate(self.classifiers):
            X_clf = X_s[:, clf.feats]
            probs = clf.predict_proba(X_clf)
            for j in range(clf.n_classes):
                final_preds[:, self.ecoc[i] == j] += probs[:, j][:, None] * clf.weight

        return final_preds / sum(clf.weight for clf in self.classifiers)
    
    def score(self, X: DataFrame, y: Series) -> float:
        preds = self.predict(X)
        return accuracy_score(y.values, preds)

@typechecked
class StackedECOC(QuantumECOC):
    """
    Stacked Quantum Ensemble that extends QuantumECOC by adding a meta-learner on top of the quantum base learners.
    """
    def __init__(
        self,
        meta_learner = None,
        n_learners: int | None = None,
        templates: int | str | list[int | str] = "default",
        feature_density: int = 1,
        ecoc_depth: int = 2,
        device: str = "default.qubit",
        **kwargs
    ):
        super().__init__(n_learners=n_learners, templates=templates, feature_density=feature_density, ecoc_depth=ecoc_depth, device=device, **kwargs)
        self.meta_learner = meta_learner if meta_learner is not None else SVC(kernel="rbf", random_state=2, probability=True)

    def reset_ensemble(self):
        for clf in self.classifiers:
            clf.reset_model()

    def fit(
        self,
        X: DataFrame,
        y: Series,
        X_test: DataFrame | None = None,
        y_test: Series | None = None,
        k_folds: int = 5,
        bagging: tuple[float, int, int] | None = (0.5, 512, 2048),
        plot: bool = False,
        verbose: bool = False,
        **fit_params
    ):
        start_time = time.time()
        self.initialise_ensemble(X, y, verbose=verbose)

        class_samples = y.value_counts()
        k_folds = min(k_folds, int(class_samples.min()))
        if k_folds > 1:
            X_meta = []
            y_meta = []
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=2)

            for f, (train_index, val_index) in enumerate(skf.split(X, y)):
                if verbose:
                    print(f"\nFold {f + 1}/{k_folds}")
                
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                fold_scaler = MinMaxScaler(feature_range=self.scaler.feature_range)
                X_train = fold_scaler.fit_transform(X_train)
                X_val = fold_scaler.transform(X_val)

                y_train = y_train.map(self.label_to_int)
                y_val = y_val.map(self.label_to_int)

                self.train_ensemble(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    fold=f,
                    bagging=bagging,
                    plot=False,
                    verbose=False,
                    **fit_params,
                )

                X_fold = np.array([clf.val_probs for clf in self.classifiers]).T
                X_fold = X_fold[1:, :].transpose((1, 0, 2)).reshape(len(X_val), -1)
                
                X_meta.extend(X_fold)
                y_meta.extend(y_val)

                self.reset_ensemble()

            if verbose:
                print("\nTraining meta-learner...")
            self.meta_learner.fit(np.array(X_meta), np.array(y_meta))

            if verbose:
                print("\nTraining final ensemble on full dataset...")

            X_s = self.scaler.fit_transform(X)
            X_test_s = self.scaler.transform(X_test) if X_test is not None else None

            y_m = y.map(self.label_to_int)
            y_test_m = y_test.map(self.label_to_int) if y_test is not None else None

            self.train_ensemble(
                X_s,
                y_m,
                X_test_s,
                y_test_m,
                bagging=bagging,
                plot=plot,
                verbose=verbose,
                **fit_params,
            )
        else:
            if verbose:
                print(f"\nNot enough samples for {k_folds} folds. Training on a single train/validation split...")

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
            
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)

            y_train = y_train.map(self.label_to_int)
            y_val = y_val.map(self.label_to_int)

            self.train_ensemble(
                X_train,
                y_train,
                X_val,
                y_val,
                bagging=bagging,
                plot=plot,
                verbose=verbose,
                **fit_params,
            )

            X_meta = np.array([clf.val_probs for clf in self.classifiers]).T
            X_meta = X_meta[1:, :].transpose((1, 0, 2)).reshape(len(X_val), -1)

            if verbose:
                print("\nTraining meta-learner...")
            self.meta_learner.fit(X_meta, np.array(y_val))

        self.training_time = TimeInt(time.time() - start_time)
        if verbose:
            print(f"\nTotal training time after fitting {self.n_learners} classifiers across {k_folds} folds: {self.training_time}")

    def predict(self, X: DataFrame) -> np.ndarray:
        X_s = self.scaler.transform(X)
        X_meta = np.array([
            clf.predict_proba(
                X_s[:, clf.feats]
            ) for clf in self.classifiers
        ]).T
        X_meta = X_meta[1:, :].transpose((1, 0, 2)).reshape(len(X), -1)
        return np.array([self.int_to_label[pred] for pred in self.meta_learner.predict(X_meta)])
    
    def predict_proba(self, X: DataFrame) -> np.ndarray:
        X_s = self.scaler.transform(X)
        X_meta = np.array([
            clf.predict_proba(
                X_s[:, clf.feats]
            ) for clf in self.classifiers
        ]).T
        X_meta = X_meta[1:, :].transpose((1, 0, 2)).reshape(len(X), -1)

        return self.meta_learner.predict_proba(X_meta)
    
    def score(self, X: DataFrame, y: Series) -> float:
        preds = self.predict(X)
        return accuracy_score(y.values, preds)

class CoherentECOC(QuantumECOC):
    """
    Measurement Free Quantum Ensemble that extends QuantumECOC by combining the outputs of the base learners without collapsing their quantum states.
    """
    def __init__(
        self,
        meta_layers: int | list[int] = 2,
        meta_design: str = "full",
        n_learners: int | None = None,
        templates: int | str | list[int | str] = 0,
        ecoc_depth: int = 2,
        device: str = "default.qubit",
        **kwargs
    ):
        super().__init__(n_learners=n_learners, templates=templates, ecoc_depth=ecoc_depth, device=device, **kwargs)
        if isinstance(meta_layers, int):
            self.meta_layers = [meta_layers]
        else:
            self.meta_layers = meta_layers
        self.meta_design = meta_design

    def reset_ensemble(self):
        for i in range(len(self.classifiers)):
            clf = VQC(self.qml_device, len(set(self.ecoc[i])), self.templates[i]).to(self.cuda_device)
            self.ecoc[i] = self.ecoc[i]
            clf.feats = self.feat_map[i]
            self.classifiers[i] = clf

    def get_base_weights(self, idx: int) -> torch.Tensor:
        return self.classifiers[idx].qlayer.weights.detach().clone()
    
    def get_input(self, X: np.ndarray) -> np.ndarray:
        inputs = [X[:, clf.feats] for clf in self.classifiers]
        return np.concatenate(inputs, axis=1)

    def build_coherent_circuit(self):
        total_qubits = 0
        main_wires = []
        for clf in self.classifiers:
            main_wires.append(total_qubits)
            total_qubits += clf.n_qubits
        
        if self.meta_design == "full":
            self.wires = list(range(total_qubits))
        elif self.meta_design == "main":
            self.wires = main_wires

        wire_maps = []
        n_features_list = []
        for i, clf in enumerate(self.classifiers):
            wire_map = {
                w: w + main_wires[i]
                for w in range(clf.n_qubits)
            }
            wire_maps.append(wire_map)

            n_features_list.append(clf.n_features)

        coherent_device = qml.device(self.qml_device, wires=total_qubits)
        @qml.qnode(coherent_device, interface="torch", diff_method="best")
        def coherent_circuit(inputs, weights):
            prev_idx = 0

            for i, clf in enumerate(self.classifiers):
                n_features_i = n_features_list[i]

                inputs_i = inputs[..., prev_idx:prev_idx + n_features_i]

                weights_i = self.get_base_weights(i).to(inputs.device)

                tape = qml.tape.make_qscript(clf.circuit.func)(inputs_i, weights_i)
                for op in tape.operations:
                    qml.apply(qml.map_wires(op, wire_maps[i]))

                prev_idx += n_features_i

            for i in range(len(self.meta_layers)):
                entangling_block(weights, 1, self.meta_layers[i], i, len(self.wires), [qml.RX, qml.RZ], qml.CZ, wires=self.wires)

            return qml.probs(wires=self.wires)

        n_features = sum(n_features_list)

        weight_shapes = {
            "weights": (len(self.meta_layers), len(self.wires), 2)
        }

        coherent_layer = qml.qnn.TorchLayer(
            coherent_circuit,
            weight_shapes
        )

        self.coherent_vqc = VQC(
            coherent_device,
            n_classes=self.n_classes,
            template="manual",
            qlayer=coherent_layer,
            circuit=coherent_circuit,
            weight_shapes=weight_shapes,
            n_features=n_features
        ).to(self.cuda_device)

    def fit(
        self,
        X: DataFrame,
        y: Series,
        X_test: DataFrame | None = None,
        y_test: Series | None = None,
        k_folds: int = 5,
        bagging: tuple[float, int, int] | None = (0.5, 512, 2048),
        plot: bool = False,
        verbose: bool = False,
        **fit_params
    ):
        start_time = time.time()
        self.initialise_ensemble(X, y, verbose=verbose)
        self.build_coherent_circuit()

        class_samples = y.value_counts()
        k_folds = min(k_folds, int(class_samples.min()))
        if k_folds > 1:
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=2)

            X_main, X_tune, y_main, y_tune = train_test_split(X, y, test_size=0.1, random_state=2, stratify=y)

            for f, (train_index, val_index) in enumerate(skf.split(X_main, y_main)):
                if verbose:
                    print(f"\nFold {f + 1}/{k_folds}")
                
                X_train, X_val = X_main.iloc[train_index], X_main.iloc[val_index]
                y_train, y_val = y_main.iloc[train_index], y_main.iloc[val_index]

                fold_scaler = MinMaxScaler(feature_range=self.scaler.feature_range)
                X_train = fold_scaler.fit_transform(X_train)
                X_val = fold_scaler.transform(X_val)

                y_train = y_train.map(self.label_to_int)
                y_val = y_val.map(self.label_to_int)

                self.train_ensemble(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    fold=f,
                    bagging=bagging,
                    plot=False,
                    verbose=False,
                    **fit_params,
                )

                if verbose:
                    print(f"\nTraining final circuit on fold {f + 1}")
                self.coherent_vqc.fit(self.get_input(X_val), y_val.values, plot=plot, verbose=verbose, **fit_params)

                self.reset_ensemble()

            if verbose:
                print("\nTraining final ensemble on full dataset...")

            X_s = self.scaler.fit_transform(X_main)
            X_test_s = self.scaler.transform(X_test) if X_test is not None else None

            y_m = y_main.map(self.label_to_int)
            y_test_m = y_test.map(self.label_to_int) if y_test is not None else None

            self.train_ensemble(
                X_s,
                y_m,
                X_test_s,
                y_test_m,
                bagging=bagging,
                plot=plot,
                verbose=verbose,
                **fit_params,
            )

            X_tune_s = self.scaler.transform(X_tune)
            y_tune_m = y_tune.map(self.label_to_int)

            if verbose:
                print("\nFine tuning final circuit...")
            self.coherent_vqc.fit(self.get_input(X_tune_s), y_tune_m.values, self.get_input(X_test_s), y_test_m.values, plot=plot, verbose=verbose, **fit_params)
        else:
            if verbose:
                print(f"\nTraining on a single train/validation split...")

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
            
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)

            y_train = y_train.map(self.label_to_int)
            y_val = y_val.map(self.label_to_int)

            self.train_ensemble(
                X_train,
                y_train,
                X_val,
                y_val,
                bagging=bagging,
                plot=plot,
                verbose=verbose,
                **fit_params,
            )

            X_meta = self.get_input(X_val)

            if verbose:
                print("\nTraining meta-learner...")
            self.coherent_vqc.fit(X_meta, y_val.values, plot=plot, verbose=verbose, **fit_params)

        self.training_time = TimeInt(time.time() - start_time)
        if verbose:
            print(f"\nTotal training time after fitting {self.n_learners} classifiers across {k_folds} folds: {self.training_time}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.coherent_vqc.eval()
        with torch.no_grad():
            X_t = torch.tensor(self.get_input(self.scaler.transform(X)), dtype=torch.float32).to(self.cuda_device)
            return self.coherent_vqc(X_t.to(self.cuda_device)).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict_proba(X).argmax(axis=1)
        return np.array([self.int_to_label[pred] for pred in preds])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return accuracy_score(y, preds)

if __name__ == "__main__":
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
    # X = pd.read_csv(f"datasets/1/image-segmentation.csv")
    y = X['target']
    X = X.drop('target', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

    # scaler = MinMaxScaler(feature_range=(0, np.pi))
    # X_tr = scaler.fit_transform(X_train)
    # X_te = scaler.transform(X_test)

    # y_tr = y_train.map({label: idx for idx, label in enumerate(sorted(y.unique()))}).values
    # y_te = y_test.map({label: idx for idx, label in enumerate(sorted(y.unique()))}).values

    # vqc = VQC(
    #     n_classes=len(np.unique(y_train)),
    #     feats=list(range(X_train.shape[1])),
    #     template='hea_cz_ring',
    # ).to("cpu")
    # vqc.ansatz_spec.draw_mpl(decimals=2)
    # vqc.fit(X_tr, y_tr, X_te, y_te, epochs=200, plot=False, verbose=True)
    # vqc.ansatz_spec.draw_mpl(decimals=2, weights=vqc.qlayer.weights.detach().cpu().numpy())
    # print(f"VQC Classification Report:\n{classification_report(y_te, vqc.predict(X_te), zero_division=0)}\n")
    # vqc.ansatz_spec.draw_mpl(decimals=2, weights=vqc.qlayer.weights.detach().cpu().numpy())

    # ens = QuantumECOC(templates='hea_cz_ring').to("cpu")
    # ens.fit(X_train, y_train, X_test, y_test, epochs=100, plot=False, verbose=True)
    # preds = ens.predict(X_test)
    # print(f"QuantumECOC Classification Report:\n{classification_report(y_test, preds, zero_division=0)}\n")

    ens = StackedECOC(templates='hea_cz_ring').to("cpu")
    ens.fit(X_train, y_train, X_test, y_test, epochs=100, plot=False, verbose=True)
    preds = ens.predict(X_test)
    print(f"StackedECOC Classification Report:\n{classification_report(y_test, preds, zero_division=0)}\n")

    # dev = qml.device("default.qubit", wires=4)
    # metaVQC = VQC(dev, len(np.unique(y_train)), template='meta').to("cpu")
    # ens = StackedECOC(meta_learner=metaVQC).to("cpu")
    # ens.fit(X_train, y_train, X_test, y_test, epochs=200, plot=False, verbose=False)
    # preds = ens.predict(X_test)
    # print(f"StackedECOC Classification Report:\n{classification_report(y_test, preds, zero_division=0)}\n")

    # plt.figure(figsize=(10, 6))
    # for j, clf in enumerate(ens.classifiers):
    #     plt.plot(clf.val_losses, label=f"Classifier {j + 1} (Code: {ens.ecoc[j]})")
    # plt.xlabel("Epoch")
    # plt.ylabel("Validation Loss")
    # plt.title("Validation Loss for Each Classifier")
    # plt.xlim(0, max(1, max(len(clf.val_losses) for clf in ens.classifiers) - 1))
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # model = CoherentECOC(meta_layers=[1],  templates=1).to("cpu")
    # model.fit(X_train, y_train, X_test, y_test, k_folds=5, epochs=200, plot=False, verbose=True)
    # print(f"CoherentECOC Classification Report:\n{classification_report(y_test, model.predict(X_test), zero_division=0)}\n")