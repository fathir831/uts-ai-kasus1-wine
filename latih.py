"""latih.py - pelatihan, pemilihan model (CV di train), evaluasi test 1x, simpan joblib."""
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fitur import FITUR_ASLI, TARGET, RasioFitur, audit_data, muat_data

# Sembunyikan FutureWarning SVC(probability=True) di proses worker CV
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

SEED = 42
DIR_MODEL = Path("models")


def buat_pipeline(mode, estimator):
    # Scaler berada DI DALAM pipeline -> di-fit hanya pada data train (per fold saat CV)
    return Pipeline([("fitur", RasioFitur(mode=mode)),
                     ("scaler", StandardScaler()),
                     ("model", estimator)])


def kandidat_model():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "SVM (RBF)": SVC(probability=True, random_state=SEED),
    }


def main():
    DIR_MODEL.mkdir(exist_ok=True)
    df = muat_data()
    print("== AUDIT DATA ==")
    for k, v in audit_data(df).items():
        print(f"{k}: {v}")

    X, y = df[FITUR_ASLI], df[TARGET]
    # 1) SPLIT DULU, sebelum scaling/rasio/apa pun
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"\nTrain={len(X_tr)}  Test={len(X_te)}")

    # 2) Pemilihan model & varian fitur: CV hanya di data train
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    hasil = []
    for mode in ["asli", "asli+rasio", "rasio_ringkas"]:
        for nama, est in kandidat_model().items():
            skor = cross_val_score(buat_pipeline(mode, est), X_tr, y_tr,
                                   cv=cv, scoring="f1_macro", n_jobs=-1)
            hasil.append({"fitur": mode, "model": nama,
                          "cv_f1_macro_mean": skor.mean(), "cv_f1_macro_std": skor.std()})
    tabel = pd.DataFrame(hasil).sort_values("cv_f1_macro_mean", ascending=False)
    print("\n== HASIL CV (f1_macro, data train) ==")
    print(tabel.round(4).to_string(index=False))

    terbaik = tabel.iloc[0]
    # Jika selisih <= 0.005 dengan varian 'asli', pilih yang lebih sederhana
    kandidat_sederhana = tabel[(tabel.fitur == "asli") & (tabel.model == terbaik.model)].iloc[0]
    if terbaik.cv_f1_macro_mean - kandidat_sederhana.cv_f1_macro_mean <= 0.005:
        terbaik = kandidat_sederhana
    print(f"\nTerpilih: {terbaik.model} | fitur={terbaik.fitur}")

    # 3) Fit final di seluruh train, evaluasi test SATU kali
    pipe = buat_pipeline(terbaik.fitur, kandidat_model()[terbaik.model]).fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    kelas = list(pipe.classes_)
    metrik = {
        "accuracy": accuracy_score(y_te, pred),
        "f1_macro": f1_score(y_te, pred, average="macro"),
        "precision_macro": precision_score(y_te, pred, average="macro"),
        "recall_macro": recall_score(y_te, pred, average="macro"),
    }
    cm = confusion_matrix(y_te, pred, labels=kelas)
    print("\n== EVALUASI TEST (1x) ==")
    print({k: round(v, 4) for k, v in metrik.items()})
    print(pd.DataFrame(cm, index=kelas, columns=kelas))
    print(classification_report(y_te, pred, digits=3))

    # 4) Simpan pipeline + metadata untuk Streamlit
    metadata = {
        "fitur_input": FITUR_ASLI,
        "kelas": kelas,
        "model_terpilih": terbaik.model,
        "mode_fitur": terbaik.fitur,
        "rentang_fitur": {c: [float(X_tr[c].min()), float(X_tr[c].max()),
                              float(X_tr[c].median())] for c in FITUR_ASLI},
        "metrik_test": metrik,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_te, pred, output_dict=True),
        "hasil_cv": tabel.to_dict(orient="records"),
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    joblib.dump({"pipeline": pipe, "metadata": metadata}, DIR_MODEL / "wine_pipeline.joblib")
    (DIR_MODEL / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nTersimpan: models/wine_pipeline.joblib, models/metadata.json")


if __name__ == "__main__":
    main()
