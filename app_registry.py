import os
import tempfile
from flask import Flask, request, render_template_string
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from nbformat import versions
from sklearn.naive_bayes import GaussianNB

APP_HTML = """
<h2>Fake News Detection - Registered Models</h2>
<form method="post">
    <label for="model_name">Choose model family:</label>
    <select name="model_name" onchange="this.form.submit()">
        {% for m in models %}
            <option value="{{m}}" {% if m==selected_model %}selected{% endif %}>{{m}}</option>
        {% endfor %}
    </select>
    <br/>
    <label for="model_version">Choose version (default is Production):</label>
    <select name="model_version" onchange="this.form.submit()">
        {% for v in versions %}
            <option value="{{v.version}}"
                {% if v.version|string == selected_version|string %}selected{% endif %}>
                v{{v.version}} [{{v.stage}}]
            </option>
        {% endfor %}
    </select>
    <br/>
    <p>Loaded model: <b>{{selected_model}}</b> (version: <b>{{version}}</b>, stage: <b>{{stage}}</b>)</p>
    <textarea name="text" rows="6" cols="80" placeholder="Paste news headline or text here...">{{text or ''}}</textarea><br/>
    <button type="submit">Predict</button>
</form>
{% if pred is not none %}
    <h3>Prediction: <span style="color:{{ 'red' if pred=='Fake' else 'green' }}">{{ pred }}</span></h3>
    <p>Using model URI: {{model_uri}}</p>
{% endif %}
"""

app = Flask(__name__)
DEFAULT_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "FakeNewsModels")

# create a client at module import so we can list models
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
print("Using default MLflow tracking URI: http://localhost:5000")
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# simple in-memory cache: (model_name, version) -> (model, vec, model_uri, meta)
MODEL_CACHE = {}


def pick_latest_model_version(client: MlflowClient, name: str):
    # Prefer Production, then Staging, otherwise return latest version by version number
    try:
        versions_prod = client.get_model_version_by_alias(name, alias="Production")
        print("Alias and version", versions_prod.aliases, versions_prod.version)
        if versions_prod:
            mv = versions_prod
            return mv.version, mv.aliases, mv
    except Exception as e:
        print("No Production version found for model:", name)
        print("Error:",e)
    try:
        versions_staging = client.get_model_version_by_alias(name, alias="Staging")
        print(versions_staging.aliases)
        if versions_staging:
            mv = versions_staging
            return mv.version, mv.aliases, mv
    except Exception as e:
        print("No Staging version found for model:", name)
        print("Error:",e)
    # fallback: list all versions and pick the highest
    try:
        print("--Looking for all versions for model:", name)
        all_versions = client.search_model_versions(f"name='{name}'")
        print("--All verssions:", all_versions)
        if not all_versions:
            return None, None, None
        # sort by numeric version
        all_versions_sorted = sorted(all_versions, key=lambda v: int(v.version), reverse=True)
        mv = all_versions_sorted[0]
        return mv.version, mv.aliases if mv.aliases else None, mv
    except Exception as e:
        print("Error:",e)
        return None, None, None


def find_vectorizer_path_for_run(client: MlflowClient, run_id: str):
    # Recursively search artifacts for a file with 'vectorizer' in its name
    def walk(path):
        for it in client.list_artifacts(run_id, path=path):
            if it.is_dir:
                r = walk(it.path)
                if r:
                    return r
            else:
                if 'vectorizer' in it.path and it.path.endswith('.joblib'):
                    return it.path
        return None
    return walk("")


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


def load_model_and_vectorizer():
    # kept for backward compatibility but prefer load_model_and_vectorizer_for
    return load_model_and_vectorizer_for(DEFAULT_MODEL_NAME)


def load_model_and_vectorizer_for(model_name, version, stage=None):
    # check cache first (version will be resolved below)
    # version, stage, mv = pick_latest_model_version(client, model_name)

    mv = None
    if stage is not None and len(stage) > 0:
        try:
            mv = client.get_model_version_by_alias(model_name, alias=stage)
        except Exception:
            mv = None
    else:
        try:
            mv = client.get_model_version(model_name, version)
        except Exception:
            mv = None

    if version is None:
        return None, None, None, None
    cache_key = (model_name, version)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key][0], MODEL_CACHE[cache_key][1], MODEL_CACHE[cache_key][2], MODEL_CACHE[cache_key][3]

    if stage and len(stage) > 0:
        # MLflow alias
        alias = stage
        model_uri = f"models:/{model_name}@{alias}"
    else:
        # numeric version
        model_uri = f"models:/{model_name}/{version}"

    model = None
    # prefer sklearn flavor loader for raw estimator
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        try:
            model = mlflow.pyfunc.load_model(model_uri)
        except Exception:
            model = None

    vec = None
    run_id = None
    if mv is not None:
        run_id = getattr(mv, 'run_id', None) or getattr(mv, 'source_run_id', None)
        if not run_id:
            src = getattr(mv, 'source', None)
            if src and src.startswith("runs:/"):
                parts = src.split("/")
                if len(parts) >= 2:
                    run_id = parts[1]
    if run_id:
        vec_path = find_vectorizer_path_for_run(client, run_id)
        if vec_path:
            tmp = tempfile.mkdtemp()
            local = client.download_artifacts(run_id, vec_path, dst_path=tmp)
            try:
                vec = joblib.load(local)
            except Exception:
                vec = None

    meta = {'version': version, 'stage': stage, 'run_id': run_id}
    MODEL_CACHE[cache_key] = (model, vec, model_uri, meta)
    return model, vec, model_uri, meta


def list_model_versions(name: str):
    """Return list of versions for a registered model name with stage and version."""
    vers = client.search_model_versions(f"name='{name}'")
    vers = sorted(vers, key=lambda v: int(v.version), reverse=True)
    out = []
    for i, v in enumerate(vers):
        try:
            aliases = v.aliases
        except:
            aliases = []
        if len(aliases) == 0 and i != 0:
            continue
        out.append({
            "version": v.version,
            "stage": ",".join(aliases) if aliases else "nones",
            "raw": v
        })
    print("Output versions:", out)
    return sorted(out, key=lambda x: int(x["version"]), reverse=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    models = []
    try:
        client = MlflowClient()
        # print("client:", client.search_registered_models())
        regs = client.search_registered_models()
        # list all registered models so we pick up per-algorithm names created by train.py
        models = [r.name for r in regs]
        print("regs:", models)
    except Exception as e:
        models = [DEFAULT_MODEL_NAME]
        print("Error listing registered models:", e)
        pass
    print("Available models:", models)
    # prefer the form selection, otherwise pick a model that has a Production version if available
    form_selected = request.form.get('model_name')
    if form_selected:
        selected_model = form_selected
    else:
        # find any registered model with Production stage
        selected_model = None
        for m in models:
            try:
                ver, alias, mv = pick_latest_model_version(client, m)
                if alias == 'Production':
                    selected_model = m
                    break
            except Exception:
                continue
        if not selected_model:
            selected_model = models[0] if models else DEFAULT_MODEL_NAME
    versions = list_model_versions(selected_model)
    # pick default version: prefer Production, else first
    default_version = None
    for v in versions:
        if v['stage'] == 'Production':
            default_version = v['version']
            break
    if default_version is None and versions:
        default_version = versions[0]['version']
    # If the form provided a model_version, validate it against the versions for
    # the currently selected model. This prevents the case where changing the
    # model_name dropdown submits the previous model_version value (from the
    # prior model) which doesn't exist for the new model and results in
    # "ModelNotFound".
    raw_selected_version = request.form.get('model_version')
    if raw_selected_version and any(str(v['version']) == str(raw_selected_version) for v in versions):
        selected_version = str(raw_selected_version)
    else:
        selected_version = str(default_version) if default_version is not None else (str(versions[0]['version']) if versions else None)
    alias_for_selected = None
    for v in versions:
        if str(v['version']) == selected_version:
            if v['stage'] and len(v['stage']) > 0 and v['stage'] != 'nones':
                alias_for_selected = v['stage']   # take first alias
            break
    model, vec, model_uri, meta = load_model_and_vectorizer_for(selected_model, selected_version, alias_for_selected)
    # if user selected a specific version, reload that specific version
    if selected_version and meta and str(meta.get('version')) != str(selected_version):
        # attempt to load exact version
        try:
            mv = client.get_model_version(selected_model, selected_version)
            model_uri = f"models:/{selected_model}/{selected_version}"
            try:
                model = mlflow.sklearn.load_model(model_uri)
            except Exception:
                model = mlflow.pyfunc.load_model(model_uri)
            run_id = _extract_run_id_from_model_version(mv)
            vec = None
            if run_id:
                vec_path = find_vectorizer_path_for_run(client, run_id)
                if vec_path:
                    tmp = tempfile.mkdtemp()
                    local = client.download_artifacts(run_id, vec_path, dst_path=tmp)
                    try:
                        vec = joblib.load(local)
                    except Exception:
                        vec = None
            meta = {'version': str(selected_version), 'stage': getattr(mv, 'current_stage', getattr(mv, 'stage', None)), 'run_id': run_id}
        except Exception:
            pass
    pred = None
    text = ''
    if request.method == 'POST':
        text = request.form.get('text','').strip()
        if model is None:
            pred = 'ModelNotFound sorry'
        else:
            X = [text]
            try:
                # if we have a vectorizer, use it
                if vec is not None:
                    X_t = vec.transform(X)
                    # if underlying raw estimator is GaussianNB, convert to dense
                    try:
                        if model.__class__.__name__ == 'GaussianNB':
                            X_t = X_t.toarray()
                    except Exception:
                        pass
                    yhat = model.predict(X_t)
                else:
                    yhat = model.predict(X)
                pred = 'Real' if int(yhat[0]) == 1 else 'Fake'
            except Exception as e:
                pred = f'Error: {e}'

    return render_template_string(APP_HTML, models=models, selected_model=selected_model, versions=versions, selected_version=selected_version,
                                  version=meta.get('version') if meta else None, stage=meta.get('stage') if meta else None, model_uri=model_uri, pred=pred, text=text)


if __name__ == '__main__':
    # prefer env var but do not force it
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    print("Using default MLflow tracking URI: http://localhost:5000")
    mlflow.set_tracking_uri("http://localhost:5000")
    app.run(host='127.0.0.1', port=5001)
