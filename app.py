import os
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, roc_auc_score, confusion_matrix


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "predictions_log.csv")


@st.cache_data
def load_local_dataset(path="Term_Project_Dataset_20K.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def preprocess(df, target_cols=["final_grade", "pass_fail", "final_score"]):
    df = df.copy()
    df = df.dropna(how="all")
    # Fill simple missing values (numeric with median, categorical with mode)
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(exclude=[np.number]).columns:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "")

    targets = {col: None for col in target_cols if col in df.columns}

    # Separate X and y for each available target
    X = df.drop(columns=[c for c in target_cols if c in df.columns], errors="ignore")

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Return encoded features and available targets (as Series)
    for col in list(targets.keys()):
        targets[col] = df[col].copy()

    return X_encoded, targets


def train_classifier(X, y, model_path):
    le = None
    y_train = y
    # If y is string/object, label encode
    if y.dtype == object or y.dtype.name == 'category':
        le = LabelEncoder()
        y_train = le.fit_transform(y.astype(str))

    X_train, X_val, y_tr, y_val = train_test_split(X, y_train, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(class_weight="balanced", random_state=42)
    clf.fit(X_train, y_tr)

    # Save model and label encoder if any
    # determine saved classes (from label encoder if present)
    classes = le.classes_.tolist() if le is not None else list(getattr(clf, "classes_", []))
    joblib.dump({"model": clf, "le": le, "features": X.columns.tolist(), "classes": (le.classes_.tolist() if le is not None else list(clf.classes_))}, "models/pass_fail_rf.joblib")

    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return clf, le, acc, X_train


def train_regressor(X, y, model_path):
    X_train, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_tr)

    # Save model
    joblib.dump({"model": reg, "features": X.columns.tolist()}, model_path)

    preds = reg.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return reg, rmse, X_train


def load_model_file(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def align_features(X_input, feature_list):
    # Ensure input has same columns as training features
    X = X_input.copy()
    for f in feature_list:
        if f not in X.columns:
            X[f] = 0
    # Drop extra
    X = X[feature_list]
    return X


def compute_risk(pass_prob, row, columns):
    # pass_prob = probability of passing (class 1)
    risk = "Safe"
    reasons = []
    # If low pass probability -> at risk
    if pass_prob < 0.5:
        risk = "High Risk"
        reasons.append(f"Low pass probability ({pass_prob:.2f})")
    elif pass_prob < 0.75:
        risk = "At Risk"
        reasons.append(f"Moderate pass probability ({pass_prob:.2f})")
    # Look at attendance / study hours / stress if available
    if "attendance" in columns:
        att = row.get("attendance", None)
        if att is not None and att < 75:
            reasons.append(f"Low attendance ({att}%) — improve >75%")
            if risk == "Safe":
                risk = "At Risk"
    if "study_hours" in columns:
        sh = row.get("study_hours", None)
        if sh is not None and sh < 5:
            reasons.append(f"Low study hours ({sh}/week) — aim for 8–12")
            if risk == "Safe":
                risk = "At Risk"
    if "stress_level" in columns:
        sl = row.get("stress_level", None)
        if sl is not None and sl > 7:
            reasons.append(f"High stress level ({sl}) — consider counseling/sleep")
            if risk == "Safe":
                risk = "At Risk"
    return risk, reasons


def show_shap(clf, X_train, X_row):
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_row)

        # For classifiers shap_values is a list (one array per class)
        st.write("SHAP explanation (bar) for the prediction")
        plt.figure(figsize=(8, 4))
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap.summary_plot(shap_values[1], X_row, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_row, plot_type="bar", show=False)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"SHAP failed: {e}")


def plot_feature_importance(clf, features, top_n=15):
    try:
        imp = clf.feature_importances_
        inds = np.argsort(imp)[-top_n:][::-1]
        names = [features[i] for i in inds]
        vals = imp[inds]
        plt.figure(figsize=(8, 4))
        sns.barplot(x=vals, y=names, palette="viridis")
        plt.title("Feature importances (top {})".format(top_n))
        st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"Feature importance plot failed: {e}")


def show_trends(df):
    st.subheader("Trends")
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(cols) >= 2:
        x = st.selectbox("X axis", cols, index=0)
        y = st.selectbox("Y axis", cols, index=1)
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=df[x], y=df[y], alpha=0.6)
        sns.regplot(x=df[x], y=df[y], scatter=False, color="red")
        plt.xlabel(x); plt.ylabel(y)
        st.pyplot(plt.gcf())
    else:
        st.info("Not enough numeric columns for trends.")


def append_prediction_log(row: dict):
    import csv
    header = list(row.keys())
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_prediction_log(n=100):
    import pandas as pd
    if not os.path.exists(LOG_PATH):
        return None
    df_log = pd.read_csv(LOG_PATH)
    return df_log.tail(n)


# --- Model comparison UI ---
def model_comparison_ui(df):
    st.header("Model Comparison (pass_fail)")
    if df is None:
        st.warning("Load a dataset first (upload CSV or use local dataset).")
        return

    if "pass_fail" not in df.columns:
        st.error("`pass_fail` column not found in dataset — model comparison needs that binary target.")
        return

    with st.expander("Settings"):
        test_size = st.slider("Validation size (%)", 5, 40, 20)
        run_compare = st.button("Run Model Comparison")
        selected = st.multiselect("Models to include",
                                 ["Logistic Regression", "Random Forest", "Gradient Boosting"],
                                 default=["Logistic Regression", "Random Forest", "Gradient Boosting"])

    if not run_compare:
        return

    # Preprocess same as app: drop target then one-hot
    X = df.drop(columns=["final_grade", "pass_fail", "final_score"], errors="ignore")
    X = pd.get_dummies(X, drop_first=True)
    y_raw = df["pass_fail"].astype(str).fillna("Fail")  # fallback
    le_local = LabelEncoder()
    y = le_local.fit_transform(y_raw)

    # Train / val split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y)

    models = {}
    if "Logistic Regression" in selected:
        models["Logistic Regression"] = LogisticRegression(max_iter=1000, class_weight="balanced")
    if "Random Forest" in selected:
        models["Random Forest"] = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    if "Gradient Boosting" in selected:
        models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=200, random_state=42)

    results = {}
    st.write("Training and evaluating selected models (may take a moment)...")
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_val, y_pred)
        roc_auc = None
        fpr, tpr, th = None, None, None
        if hasattr(mdl, "predict_proba"):
            prob = mdl.predict_proba(X_val)[:, 1]
            try:
                roc_auc = roc_auc_score(y_val, prob)
                fpr, tpr, th = roc_curve(y_val, prob)
            except Exception:
                roc_auc = None
        results[name] = {"model": mdl, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm, "roc_auc": roc_auc, "fpr": fpr, "tpr": tpr}

    # Summary table
    summary = []
    for name, r in results.items():
        summary.append({"Model": name, "Accuracy": r["acc"], "Precision": r["prec"], "Recall": r["rec"], "F1": r["f1"], "ROC_AUC": r["roc_auc"]})
    st.subheader("Comparison Summary")
    st.table(pd.DataFrame(summary).sort_values("F1", ascending=False).reset_index(drop=True))

    # ROC plot
    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(6,4))
    any_roc = False
    for name, r in results.items():
        if r["fpr"] is not None:
            ax.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['roc_auc']:.3f})")
            any_roc = True
    if any_roc:
        ax.plot([0,1],[0,1],"k--",alpha=0.3)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("ROC not available (no probability outputs).")

    # Confusion matrices
    st.subheader("Confusion Matrices")
    for name, r in results.items():
        st.write(name)
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.success("Model comparison complete. Choose the best model in your notebook or use the app to load pre-trained model files.")

def main():
    st.set_page_config(page_title="Student Performance Predictor", layout="wide")
    st.title("Student Performance Prediction — Streamlit GUI")

    st.markdown("""
    **Instructions**
    - Upload a CSV with the same columns used in your project (or use the included dataset).
    - Train models or upload trained models in `models/`.
    - Input single student data manually or upload CSV for batch predictions.
    - Use dashboard panels for explainability and recommendations.
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("Data input")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        use_local = st.checkbox("Use local dataset (Term_Project_Dataset_20K.csv)", value=True)
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")
                df = None
        elif use_local:
            df = load_local_dataset()
            if df is None:
                st.warning("Local dataset not found in repository root.")
        else:
            df = None
        if df is not None:
            st.write("Preview:")
            st.dataframe(df.head())
            st.write(f"Shape: {df.shape}")

    with col2:
        st.header("Models & Actions")
        model_pass_path = os.path.join(MODEL_DIR, "pass_fail_rf.joblib")
        model_grade_path = os.path.join(MODEL_DIR, "final_grade_rf.joblib")
        model_score_path = os.path.join(MODEL_DIR, "final_score_rf.joblib")

        if st.button("Train models (from current dataset)"):
            if df is None:
                st.error("No dataset available to train on.")
            else:
                with st.spinner("Preprocessing and training..."):
                    X, targets = preprocess(df)
                    results = {}
                    if "pass_fail" in targets and targets["pass_fail"] is not None:
                        clf, le, acc, X_tr = train_classifier(X, targets["pass_fail"], model_pass_path)
                        results["pass_fail_acc"] = acc
                    if "final_grade" in targets and targets["final_grade"] is not None:
                        clf2, le2, acc2, X_tr2 = train_classifier(X, targets["final_grade"], model_grade_path)
                        results["final_grade_acc"] = acc2
                    if "final_score" in targets and targets["final_score"] is not None:
                        reg, rmse, X_tr3 = train_regressor(X, targets["final_score"], model_score_path)
                        results["final_score_rmse"] = rmse
                st.success("Training finished")
                for k, v in results.items():
                    st.write(f"{k}: {v:.3f}")

        st.write("---")
        st.write("Prediction")
        input_mode = st.radio("Input type:", ["Manual single input", "Upload CSV for batch"])
        if input_mode == "Manual single input":
            st.write("Fill the form fields below (values taken from dataset columns when available).")
            sample = None
            if df is not None:
                sample = df.drop(columns=[c for c in ["final_grade", "pass_fail", "final_score"] if c in df.columns],
                                 errors="ignore").iloc[0:1]
            fields = {}
            if sample is not None:
                with st.form("manual_form"):
                    for col in sample.columns:
                        fields[col] = st.text_input(col, value=str(sample[col].values[0]))
                    submit = st.form_submit_button("Predict")
                if submit:
                    single = pd.DataFrame([fields])
                    for c in single.columns:
                        try:
                            single[c] = pd.to_numeric(single[c])
                        except:
                            pass
                    X_enc, _ = preprocess(pd.concat([single, single]).reset_index(drop=True))
                    X_enc = X_enc.iloc[[0]]
                    # Pass/Fail
                    mp = load_model_file(model_pass_path)
                    if mp is None:
                        st.warning("Pass/Fail model not found. Train models or upload model in /models.")
                    else:
                        clf = mp["model"]
                        feats = mp["features"]
                        X_al = align_features(X_enc, feats)
                        probs = clf.predict_proba(X_al)[0]
                        # get saved classes (prefer explicit saved classes)
                        classes = mp.get("classes") or (mp.get("le").classes_.tolist() if mp.get("le") is not None else list(clf.classes_))
                        # robustly find index for "pass"
                        pass_idx = None
                        pass_labels = {"pass","passed","p","yes","true","1"}
                        for i, c in enumerate(classes):
                            if str(c).strip().lower() in pass_labels:
                                pass_idx = i
                                break
                        if pass_idx is None:
                            try:
                                numeric = [float(x) for x in classes]
                                pass_idx = int(numeric.index(max(numeric)))
                            except Exception:
                                pass_idx = 1 if len(classes) > 1 else 0
                        pass_prob = float(probs[pass_idx])
                        pred_class = clf.predict(X_al)[0]
                        if mp.get("le") is not None:
                            pred_label = mp["le"].inverse_transform([pred_class])[0]
                        else:
                            pred_label = pred_class
                        st.subheader("Pass/Fail Prediction")
                        st.write("Predicted:", pred_label)
                        st.write("Probabilities:", {str(c): float(p) for c, p in zip(classes, probs)})

                        # Risk & recommendations
                        row_vals = single.iloc[0].to_dict()
                        risk, reasons = compute_risk(pass_prob, row_vals, single.columns.tolist())
                        st.write("Risk level:", risk)
                        if reasons:
                            st.write("Reasons / Recommendations:")
                            for r in reasons:
                                st.write("-", r)

                        st.subheader("Explainability")
                        try:
                            show_shap(clf, X_al, X_al)
                        except Exception as e:
                            st.error(f"SHAP error: {e}")

                    # Final grade / score (if models exist)
                    mg = load_model_file(model_grade_path)
                    if mg is not None:
                        clf_g = mg["model"]
                        feats_g = mg["features"]
                        Xg = align_features(X_enc, feats_g)
                        # show top probabilities
                        if hasattr(clf_g, "predict_proba"):
                            probs = clf_g.predict_proba(Xg)[0]
                            classes = mg.get("classes") or (mg.get("le").classes_.tolist() if mg.get("le") else list(clf_g.classes_))
                            # pair classes with probs
                            prob_map = {str(c): float(p) for c, p in zip(classes, probs)}
                            st.write("Grade probabilities:", prob_map)
                        pred_g = clf_g.predict(Xg)[0]
                        if mg.get("le") is not None:
                            pred_g_label = mg["le"].inverse_transform([pred_g])[0]
                        else:
                            # if classes list saved, map numeric index to label
                            classes = mg.get("classes") or list(clf_g.classes_)
                            try:
                                pred_g_label = classes[int(pred_g)]
                            except Exception:
                                pred_g_label = pred_g
                        st.subheader("Final Grade Prediction")
                        st.write(pred_g_label)
                    ms = load_model_file(model_score_path)
                    if ms is not None:
                        reg = ms["model"]
                        feats_s = ms["features"]
                        Xs = align_features(X_enc, feats_s)
                        pred_s = reg.predict(Xs)[0]
                        st.subheader("Final Score Prediction")
                        st.write(float(pred_s))

                        # Log prediction
                        log_row = {
                          "timestamp": datetime.datetime.now().isoformat(),
                          "student_id": row.get("student_id", ""),
                          "pred_pass_fail": str(pred_label),
                          "pred_prob_pass": float(pass_prob),
                          "risk": risk,
                          **{f"feat_{k}": v for k,v in single.iloc[0].to_dict().items()}
                        }
                        append_prediction_log(log_row)

        else:
            uploaded_batch = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="batch")
            if uploaded_batch is not None:
                batch_df = pd.read_csv(uploaded_batch)
                st.write("Preview of batch:")
                st.dataframe(batch_df.head())
                if st.button("Run batch prediction"):
                    X_enc, _ = preprocess(batch_df)
                    mp = load_model_file(model_pass_path)
                    if mp is None:
                        st.error("Pass/Fail model not found.")
                    else:
                        clf = mp["model"]; feats = mp["features"]
                        X_al = align_features(X_enc, feats)
                        preds = clf.predict(X_al)
                        probs = clf.predict_proba(X_al)
                        if mp.get("le") is not None:
                            preds = mp["le"].inverse_transform(preds)
                        out = batch_df.copy()
                        out["pred_pass_fail"] = preds
                        out["pred_prob_max"] = [float(max(p)) for p in probs]
                        # determine pass index once (use saved classes if available)
                        classes = mp.get("classes") or (mp.get("le").classes_.tolist() if mp.get("le") is not None else list(clf.classes_))
                        pass_idx = None
                        for i, c in enumerate(classes):
                            if str(c).strip().lower() in {"pass","passed","p","yes","true","1"}:
                                pass_idx = i
                                break
                        if pass_idx is None:
                            try:
                                numeric = [float(x) for x in classes]
                                pass_idx = int(numeric.index(max(numeric)))
                            except Exception:
                                pass_idx = 1 if len(classes) > 1 else 0
                        out["pred_prob_pass"] = [float(p[pass_idx]) for p in probs]
                        # risk & suggestions per row
                        rows_risk = []
                        for i, row in out.iterrows():
                            pp = out.loc[i, "pred_prob_pass"]
                            risk, reasons = compute_risk(pp, row.to_dict(), batch_df.columns.tolist())
                            rows_risk.append(risk)
                        out["risk_level"] = rows_risk
                        st.write("Predictions:")
                        st.dataframe(out.head())
                        csv = out.to_csv(index=False).encode('utf-8')
                        st.download_button("Download predictions CSV", csv, file_name="predictions.csv", mime="text/csv")

    # load pretrained models (from ML_Project_Final notebook)
    mp_pass = load_model_file(model_pass_path)
    mp_grade = load_model_file(model_grade_path)
    ms_score = load_model_file(model_score_path)

    st.sidebar.header("Model Status")
    st.sidebar.write("pass_fail:", "loaded" if mp_pass is not None else "missing")
    st.sidebar.write("final_grade:", "loaded" if mp_grade is not None else "missing")
    st.sidebar.write("final_score:", "loaded" if ms_score is not None else "missing")

    # Analytics & Dashboard (Data Insights)
    st.sidebar.header("Analytics & Dashboard")

    def find_first_column(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def show_data_insights(df):
        st.header("Data Insights Dashboard")
        if df is None:
            st.warning("No dataset loaded for insights.")
            return

        # copy + ensure numeric where expected
        d = df.copy()
        numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()

        # detect useful columns (try common names)
        attendance_col = find_first_column(d, ["attendance", "past_attendance_rate", "lecture_attendance_rate", "attendance_rate"])
        study_col = find_first_column(d, ["study_time_per_week", "study_hours_last_semester", "study_hours"])
        stress_col = find_first_column(d, ["stress_level", "stress"])
        score_col = "final_score" if "final_score" in d.columns else None
        grade_col = "final_grade" if "final_grade" in d.columns else None
        pass_col = "pass_fail" if "pass_fail" in d.columns else None

        # Attendance vs Final Score
        if attendance_col and score_col:
            st.subheader(f"Attendance ({attendance_col}) vs Final Score")
            fig, ax = plt.subplots(figsize=(7,4))
            sns.scatterplot(data=d, x=attendance_col, y=score_col, alpha=0.6)
            sns.regplot(data=d, x=attendance_col, y=score_col, scatter=False, color="red", ax=ax)
            ax.set_xlabel(attendance_col)
            ax.set_ylabel(score_col)
            st.pyplot(fig)
        else:
            st.info("Attendance vs Final Score: missing attendance or final_score column.")

        # Stress vs Performance
        if stress_col and score_col:
            st.subheader(f"Stress ({stress_col}) vs Final Score")
            fig, ax = plt.subplots(figsize=(7,4))
            if pass_col:
                sns.scatterplot(data=d, x=stress_col, y=score_col, hue=pass_col, alpha=0.6)
            else:
                sns.scatterplot(data=d, x=stress_col, y=score_col, alpha=0.6)
            sns.regplot(data=d, x=stress_col, y=score_col, scatter=False, color="red", ax=ax)
            ax.set_xlabel(stress_col)
            ax.set_ylabel(score_col)
            st.pyplot(fig)
        else:
            st.info("Stress vs Performance: missing stress or final_score column.")

        # Study Hours vs Grade (boxplot)
        if study_col and grade_col:
            st.subheader(f"Study Hours ({study_col}) vs Final Grade ({grade_col})")
            fig, ax = plt.subplots(figsize=(7,4))
            sns.boxplot(data=d, x=grade_col, y=study_col, palette="viridis", order=sorted(d[grade_col].dropna().unique()))
            ax.set_xlabel(grade_col)
            ax.set_ylabel(study_col)
            st.pyplot(fig)
        else:
            st.info("Study Hours vs Grade: missing study-hours or final_grade column.")

        # Histogram / count of grades
        if grade_col:
            st.subheader("Histogram / Counts of Final Grades")
            fig, ax = plt.subplots(figsize=(6,3))
            sns.countplot(data=d, x=grade_col, order=sorted(d[grade_col].dropna().unique()), palette="magma")
            ax.set_xlabel("Final Grade")
            st.pyplot(fig)
        else:
            st.info("Histogram of grades: `final_grade` column missing.")

        # Correlation heatmap (numeric features)
        st.subheader("Correlation Heatmap (numeric features)")
        if len(numeric_cols) >= 2:
            corr = d[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(9,7))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

    # Sidebar control to show insights
    if st.sidebar.button("Show Data Insights Dashboard"):
        show_data_insights(df)

    st.info("This app uses pretrained models saved in the `models/` folder. To retrain, run training in `ML_Project_Final.ipynb` and save the model files to `models/`.")


if __name__ == "__main__":
    main()

import joblib, pandas as pd, numpy as np
# 1. Inspect saved model
m = joblib.load("models/pass_fail_rf.joblib")
print("saved keys:", m.keys())
print("classes:", m.get("classes"))
print("label encoder:", bool(m.get("le")))
print("n_features_saved:", len(m.get("features", [])))

# 2. Dataset label distribution
df = pd.read_csv("Term_Project_Dataset_20K.csv")
print(df["pass_fail"].value_counts(dropna=False))
print(df["final_grade"].value_counts(dropna=False))

# 3. Feature list mismatch
saved_features = m.get("features", [])
X = pd.get_dummies(df.drop(columns=["final_grade","pass_fail","final_score"], errors="ignore"), drop_first=True)
print("model features not in current X:", [f for f in saved_features if f not in X.columns])
print("current X columns not in model:", [c for c in X.columns if c not in saved_features][:20])

# choose categorical / numeric column lists used in notebook
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
], remainder="drop")

# Fit preproc on X and save pipeline + model
preproc.fit(X)
joblib.dump(preproc, "models/preproc.joblib")
# Later when training: X_trans = preproc.transform(X); train model on that

import joblib
m = joblib.load("models/final_grade_rf.joblib")
print("keys:", m.keys())
print("has le:", bool(m.get("le")))
print("saved classes:", m.get("classes"))
print("model.classes_:", getattr(m["model"], "classes_", None))
