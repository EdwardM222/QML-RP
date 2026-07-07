from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
import os
import pandas as pd
import numpy as np
from QuECOC import QuantumECOC, StackedECOC, VQC, TimeInt
import traceback
from sklearn.model_selection import ParameterGrid

DEVICE = "cpu"

def train_model(name, dataset, model_args, fit_args=None):
    if fit_args is None:
        fit_args = {}

    print(f"Training {name}...")

    X = pd.read_csv(dataset)
    y = X['target']
    X = X.drop('target', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

    if name.startswith("SVC"):
        model = SVC(**model_args)
        model.fit(X_train, y_train, **fit_args)
    elif name.startswith("Random Forest"):
        model = RandomForestClassifier(**model_args)
        model.fit(X_train, y_train, **fit_args)
    elif name.startswith("VQC"):
        c = len(np.unique(y_train))
        model = VQC(n_classes=c, **model_args).to(DEVICE)
        model.fit(X_train, y_train, X_test, y_test, **fit_args)
    elif name.startswith("QuantumECOC"):
        model = QuantumECOC(**model_args).to(DEVICE)
        model.fit(X_train, y_train, X_test, y_test, **fit_args)
    elif name.startswith("StackedECOC"):
        model = StackedECOC(**model_args).to(DEVICE)
        model.fit(X_train, y_train, X_test, y_test, **fit_args)
    elif name.startswith("Quantum StackedECOC"):
        c = len(np.unique(y_train))
        metaVQC = VQC.from_template(
            template='meta',
            n_classes=c,
            n_total_features=X_train.shape[1]
        ).to("cpu")
        model = StackedECOC(meta_learner=metaVQC, **model_args).to(DEVICE)
        model.fit(X_train, y_train, X_test, y_test, **fit_args)

    report = classification_report(y_test, model.predict(X_test), zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.loc['accuracy'] = [np.nan, np.nan, report_df.loc['accuracy', 'f1-score'], report_df.loc['macro avg', 'support']]

    print(f"Finished {name}.")

    return report_df, model.training_time if hasattr(model, 'training_time') else None

def search(model_name, dataset_path, param_grid, fit_args=None):
    results = []
    for params in ParameterGrid(param_grid):
        name = f"{model_name} - {params}"
        report_df, training_time = train_model(name, dataset_path, params, fit_args)
        results.append((name, report_df, training_time))

    return results

def create_search_jobs(model_name, dataset_path, param_grid, fit_args=None):
    jobs = []
    for params in ParameterGrid(param_grid):
        jobs.append((f"{model_name}", dataset_path, params, fit_args))

    return jobs



if __name__ == "__main__":
    jobs = []
    for tier in [0]:
        for dataset in sorted(os.listdir(f"datasets/{tier}")):
            path = os.path.join(f"datasets/{tier}", dataset)

            jobs.extend(create_search_jobs("SVC", path, {
                'kernel': ['rbf'],
                'class_weight': ['balanced'],
                'random_state': [2]
            }))

            jobs.extend(create_search_jobs("Random Forest", path, {
                'n_estimators': [100],
                'class_weight': ['balanced'],
                'random_state': [2]
            }))

            jobs.extend(create_search_jobs("QuantumECOC", path, {
                'templates': ["1"]
            }))

            jobs.extend(create_search_jobs("StackedECOC", path, {
                'templates': ["1"]
            }))

            jobs.extend(create_search_jobs("Quantum StackedECOC", path, {
                'templates': ["1"]
            }))

    results = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(train_model, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            job_name = futures[future]
            try:
                report_df, training_time = future.result()
                results.append((report_df, training_time))

                # Save the report to a file at the end of each job
                
            except Exception:
                print(f"Error occurred while processing {job_name}: {traceback.format_exc()}")
    final_time = TimeInt(time.time() - start_time)
    print(f"\nTotal execution time: {final_time}")

    for i, (report_df, training_time) in enumerate(results):
        name = jobs[i][0]
        dataset = jobs[i][1]
        args = jobs[i][2]

        print(f"\n --- {name} - {dataset} - {args}: {training_time} --- ")
        print(report_df.round(3).fillna(''))