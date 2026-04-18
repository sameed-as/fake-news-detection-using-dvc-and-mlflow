import argparse
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib
import mlflow.sklearn
import os
import dvc.api
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient
import io

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--load_dvc", action="store_true", help="Whether to load data using DVC API")
    p.add_argument("--no-load_dvc", action="store_false", dest="load_dvc", help="Do not load data using DVC API")
    p.set_defaults(load_dvc=False)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["title","label"])
    df["text"] = df["title"].astype(str)
    df.to_csv(".\\data\\processed\\cleaned_data.csv", index=False)
    return df

def load_data_dvc(path):
    repo = "." 
    with dvc.api.open(path=path, repo=repo) as fd:
        print("Loading data from DVC path:", fd.name)
        df = load_and_clean(fd.name)
    return df

def get_vectorizer():
    return TfidfVectorizer(max_features=10000)

def get_models(rs):
    return [RandomForestClassifier(n_estimators=200, random_state=rs), GaussianNB()]

def plot_and_log_confusion(cm, labels, out_path, run):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    mlflow.log_artifact(out_path, artifact_path=f"model_artifacts_{run}")

def plot_and_log_loss(loss_vals, out_path, run):
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(loss_vals)+1), loss_vals, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss over epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    mlflow.log_artifact(out_path, artifact_path=f"model_artifacts_{run}")


def save_artifacts_and_plots(vec, model_name, cm, loss_history):
    """Save vectorizer, plots and log artifacts to MLflow for the current active run."""
    os.makedirs("outputs", exist_ok=True)
    vec_path = f"outputs/vectorizer_{model_name}.joblib"
    joblib.dump(vec, vec_path)
    # plot & log confusion matrix
    cm_path = f"outputs/cm_{model_name}.png"
    plot_and_log_confusion(cm, labels=[0, 1], out_path=cm_path, run=model_name)
    # plot & log loss if collected
    if loss_history:
        loss_path = f"outputs/loss_{model_name}.png"
        plot_and_log_loss(loss_history, loss_path, run=model_name)
    # log artifacts directory
    mlflow.log_artifacts("outputs", artifact_path=f"model_artifacts_{model_name}")


def log_sklearn_model(model, model_name, X_train_input):
    """Log sklearn model to MLflow, trying to infer signature where possible."""
    try:
        from mlflow.models import infer_signature
        signature = infer_signature(X_train_input, model.predict(X_train_input))
        mlflow.sklearn.log_model(model, f"sklearn-model-{model_name}", signature=signature)
    except Exception:
        mlflow.sklearn.log_model(model, f"sklearn-model-{model_name}")


def register_best_model(run_id, model_name, experiment_name, registered_model_name="FakeNewsModel"):
    """Register the model for the given run and manage stages.

    Logic:
    - Register the model artifact from the given run_id.
    - If there is an existing Production version, compare its f1_score to the candidate's f1_score.
      - If candidate f1 > production f1 -> promote candidate to Production and set other versions to Staging.
      - Otherwise, set candidate's stage to Staging.
    - If there is no existing Production version, promote candidate to Production.

    Returns (registered_name, version) on success or (None, None) on failure/no-op.
    """
    client = MlflowClient()
    try:
        # get candidate f1
        try:
            run_info = client.get_run(run_id)
            candidate_f1 = float(run_info.data.metrics.get("f1_score", float("nan")))
        except Exception:
            candidate_f1 = float("nan")

        # check existing production version
        try:
            prod_versions = client.get_latest_versions(registered_model_name, stages=["Production"])
        except Exception:
            prod_versions = []

        prod_f1 = float("-inf")
        prod_mv = None
        prod_run_id = None
        if prod_versions:
            prod_mv = prod_versions[0]
            # attempt to extract run id from model version
            prod_run_id = getattr(prod_mv, 'run_id', None) or getattr(prod_mv, 'source_run_id', None)
            if not prod_run_id:
                src = getattr(prod_mv, 'source', None)
                if src and src.startswith("runs:/"):
                    parts = src.split("/")
                    if len(parts) >= 2:
                        prod_run_id = parts[1]
            if prod_run_id:
                try:
                    prod_run = client.get_run(prod_run_id)
                    prod_f1 = float(prod_run.data.metrics.get("f1_score", float("nan")))
                except Exception:
                    prod_f1 = float("-inf")

        # register candidate model
        try:
            model_uri = f"runs:/{run_id}/sklearn-model-{model_name}"
            mv = mlflow.register_model(model_uri, registered_model_name)
            print(f"Registered model '{registered_model_name}' version: {mv.version}")
        except Exception as e:
            print("Model registration failed:", e)
            try:
                client.set_tag(run_id, "model_registered", "failed_to_register")
            except Exception:
                pass
            return None, None

        # Decide stage assignment
        try:
            # candidate better than existing production -> make it Production and set others to Staging
            promote_to_prod = False
            try:
                if prod_mv is None:
                    promote_to_prod = True
                else:
                    # only promote if candidate strictly better
                    if not (np.isnan(candidate_f1) or np.isnan(prod_f1)) and candidate_f1 > prod_f1:
                        promote_to_prod = True
            except Exception:
                promote_to_prod = True

            if promote_to_prod:
                try:
                    # transition this version to Production
                    client.transition_model_version_stage(name=registered_model_name, version=mv.version, stage="Production", archive_existing_versions=False)
                    # transition any other versions to Staging (best-effort)
                    all_versions = client.search_model_versions(f"name='{registered_model_name}'")
                    for v in all_versions:
                        if v.version != mv.version:
                            try:
                                client.transition_model_version_stage(name=registered_model_name, version=v.version, stage="Staging", archive_existing_versions=False)
                            except Exception:
                                pass
                    client.set_tag(run_id, "model_registered", f"{registered_model_name}:{mv.version}")
                    client.set_tag(run_id, "model_stage", "Production")
                    print(f"Promoted version {mv.version} to Production")
                    return registered_model_name, mv.version
                except Exception as e:
                    print("Failed to promote to Production (continuing):", e)
                    try:
                        client.set_tag(run_id, "model_registered", f"{registered_model_name}:{mv.version}")
                    except Exception:
                        pass
                    return registered_model_name, mv.version
            else:
                # candidate is worse than production -> set to Staging
                try:
                    client.transition_model_version_stage(name=registered_model_name, version=mv.version, stage="Staging", archive_existing_versions=False)
                except Exception:
                    pass
                try:
                    client.set_tag(run_id, "model_registered", f"{registered_model_name}:{mv.version}")
                    client.set_tag(run_id, "model_stage", "Staging")
                except Exception:
                    pass
                print(f"Registered version {mv.version} as Staging (worse than Production)")
                return registered_model_name, mv.version
        except Exception as e:
            print("Error while setting model stages:", e)
            return registered_model_name, mv.version
    except Exception as e:
        print("Unexpected error in register_best_model:", e)
        return None, None


def register_model_only(run_id, model_name, registered_model_name="FakeNewsModel"):
    """Register the model version for the given run_id under registered_model_name without changing stages."""
    client = MlflowClient()
    try:
        model_uri = f"runs:/{run_id}/sklearn-model-{model_name}"
        mv = mlflow.register_model(model_uri, registered_model_name)
        print(f"Registered model '{registered_model_name}' version: {mv.version}")
        try:
            client.set_tag(run_id, "model_registered", f"{registered_model_name}:{mv.version}")
        except Exception:
            pass
        return registered_model_name, mv.version
    except Exception as e:
        print("Model registration failed:", e)
        try:
            client.set_tag(run_id, "model_registered", "failed_to_register")
        except Exception:
            pass
        return None, None


def _extract_run_id_from_model_version(mv):
    # try common attributes
    run_id = getattr(mv, 'run_id', None) or getattr(mv, 'source_run_id', None)
    if run_id:
        return run_id
    src = getattr(mv, 'source', None)
    if src and isinstance(src, str) and src.startswith("runs:/"):
        parts = src.split("/")
        if len(parts) >= 2:
            return parts[1]
    return None


def finalize_and_promote_best(experiment_name, registered_suffix="_model"):
    """After all runs are registered, select the single best model across registered families
    and promote it to Production; set all other registered versions to Staging.
    The selection is based on the run-level metric 'f1_score'.
    """
    client = MlflowClient()
    try:
        # list registered models and filter by suffix
        try:
            regs = client.search_registered_models()
            print("registered models:", [r.name for r in regs])
            registered_names = [r.name for r in regs if r.name.endswith(registered_suffix)]
        except Exception as e:
            print("Could not list registered models; skipping finalization.", e)
            return None

        best = {
            'registered_name': None,
            'version': None,
            'run_id': None,
            'f1': float('-inf')
        }

        # iterate over registered models and their versions
        for name in registered_names:
            try:
                versions = client.search_model_versions(f"name='{name}'")
            except Exception:
                versions = []
            for v in versions:
                run_id = _extract_run_id_from_model_version(v)
                if not run_id:
                    continue
                try:
                    run = client.get_run(run_id)
                    f1 = run.data.metrics.get('f1_score')
                    if f1 is None:
                        continue
                    f1 = float(f1)
                except Exception:
                    continue
                if f1 > best['f1']:
                    best.update({'registered_name': name, 'version': v.version, 'run_id': run_id, 'f1': f1})

        if best['registered_name'] is None:
            print("No candidate found to promote; ensure models were registered and have f1_score metrics.")
            return None

        # promote best to Production, set others to Staging
        print(f"Promoting best model {best['registered_name']} version {best['version']} (f1={best['f1']}) to Production")
        try:
            client.set_registered_model_alias(name=best['registered_name'], version=best['version'], alias="Production")
        except Exception as e:
            print("Failed to transition best model to Production:", e)
            # tag desired stage for manual promotion later
            try:
                client.set_tag(best['run_id'], 'model_desired_stage', 'Production')
            except Exception:
                pass

        # set every other version of every registered model to Staging (best-effort)
        for name in registered_names:
            try:
                versions = client.search_model_versions(f"name='{name}'")
            except Exception:
                versions = []
            latest_version = None
            if name != best['registered_name']:
                versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
                latest_version = versions[0]
            else:
                versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
                latest_version = versions[1]
            
            # for v in versions:
            v = latest_version
            print("Updating model:", v.name, v.version)
            try:
                if not (v.name == best['registered_name'] and v.version == best['version']):
                    try:
                        client.set_registered_model_alias(name=v.name, alias="Staging", version=v.version)
                        print("Set to Staging:", v.name, v.version)
                    except Exception as e:
                        print("Failed to set to Staging:", v.name, v.version)
                        print(e)
                    # tag corresponding run
                    run_id = _extract_run_id_from_model_version(v)
                    if run_id:
                        try:
                            client.set_tag(run_id, 'model_stage', 'Staging')
                        except Exception:
                            pass
            except Exception:
                pass

        # mark best run tags
        try:
            client.set_tag(best['run_id'], 'model_stage', 'Production')
        except Exception:
            pass

        print("Finalization complete: Production set and other versions moved to Staging (best-effort).")
        return best
    except Exception as e:
        print("Error during finalization/promotion:", e)
        return None

def train_model(model_name, model, X_train_t, y_train, X_test_t, y_test, vec, experiment_name):
    with mlflow.start_run(run_name=f"run_{model_name}") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("vectorizer", "TfidfVectorizer")

        # prepare data (dense for models that need it)
        if isinstance(model, GaussianNB):
            X_train_input = X_train_t.toarray()
            X_test_input = X_test_t.toarray()
        else:
            X_train_input = X_train_t
            X_test_input = X_test_t

        # If model supports partial_fit -> train with epochs and log loss over time
        loss_history = []
        classes = np.unique(y_train)
        if hasattr(model, "partial_fit"):
            # first call may need classes
            for epoch in range(1, 11):
                # use full-batch partial_fit for simplicity
                try:
                    model.partial_fit(X_train_input, y_train, classes=classes)
                except TypeError:
                    model.partial_fit(X_train_input, y_train)
                # compute loss if possible
                try:
                    probs = model.predict_proba(X_test_input)
                    loss = float(sklearn_log_loss(y_test, probs))
                except Exception:
                    loss = 1.0 - accuracy_score(y_test, model.predict(X_test_input))
                loss_history.append(loss)
                mlflow.log_metric("loss", loss, step=epoch)
                # also log intermediate f1
                f1_epoch = f1_score(y_test, model.predict(X_test_input))
                mlflow.log_metric("f1_score", float(f1_epoch), step=epoch)
            # final predictions
            preds = model.predict(X_test_input)
        else:
            # single-shot training (e.g., RandomForest)
            model.fit(X_train_input, y_train)
            try:
                probs = model.predict_proba(X_test_input)
                loss = float(sklearn_log_loss(y_test, probs))
            except Exception:
                loss = None
            preds = model.predict(X_test_input)
            if loss is not None:
                mlflow.log_metric("loss", float(loss), step=0)

        # metrics and confusion matrix
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_score", float(f1))
        mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")

        # save artifacts & plots (vectorizer, confusion, loss) and log artifacts
        save_artifacts_and_plots(vec, model_name, cm, loss_history)

        # log the sklearn model
        log_sklearn_model(model, model_name, X_train_input)

        print("Logged run to MLflow")

        # ==== compare this run to other runs and register best model per algorithm ====
        try:
            # register under a model-name specific registry entry, e.g. "randomforestclassifier_model"
            safe_name = f"{model_name}".lower().replace(' ', '_') + "_model"
            register_model_only(run_id, model_name, registered_model_name=safe_name)
        except Exception as e:
            print("Error while registering the model:", e)

        import shutil
        if os.path.exists("outputs"):
            shutil.rmtree("outputs")

if __name__ == "__main__":
    # mlflow_tracking = os.environ.get("MLFLOW_TRACKING_URI")
    # if mlflow_tracking:
    #     print(f"Using MLflow tracking URI from environment: {mlflow_tracking}")
    #     mlflow.set_tracking_uri(mlflow_tracking)
    # else:
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    print("Using default MLflow tracking URI: http://localhost:5000")
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "Fake_News_Classification"
    mlflow.set_experiment(experiment_name)

    print("Ensuring no active MLflow run...")
    if mlflow.active_run() is not None:
        print(f"Ending rogue run: {mlflow.active_run().info.run_id}")
        mlflow.end_run()
        
    args = parse_args()
    if args.load_dvc:
        print("Loading via DVC API")
        df = load_data_dvc(args.data_path)
    else:
        print("Loading via direct path")
        df = load_and_clean(args.data_path)
    # combine title + text (if text column exists)
    title = df["title"].astype(str) if "title" in df else pd.Series("", index=df.index)
    textcol = df["text"].astype(str) if "text" in df else pd.Series("", index=df.index)
    X = (title + " " + textcol).str.strip()
    y = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)

    print("Vectorizing text data...")
    vec = get_vectorizer()
    X_train_t = vec.fit_transform(X_train)
    X_test_t = vec.transform(X_test)

    models = get_models(args.random_state)

    with mlflow.start_run() as run:
        # to let nested runs have correct names
        pass
    for model in models:
        model_name = type(model).__name__
        print(f"Training model: {model_name}")
        train_model(model_name, model, X_train_t, y_train, X_test_t, y_test, vec, experiment_name)

    # after all models have been trained and registered, pick a single best across registered families
    print("Finalizing registry: selecting a single best model across registered families and promoting it to Production...")
    finalize_and_promote_best(experiment_name, registered_suffix="_model")
        