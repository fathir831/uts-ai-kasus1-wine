"""fitur.py - audit data dan feature engineering untuk Kasus 1 (Wine)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "kultivar"
FITUR_ASLI = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline",
]
# Fitur rasio: pasangan fitur yang saling terkait secara kimia / korelasi tinggi
RASIO = {
    "rasio_flavanoid_fenol": ("flavanoids", "total_phenols"),            # r=0.86
    "rasio_nonflav_flav": ("nonflavanoid_phenols", "flavanoids"),
    "rasio_od280_flavanoid": ("od280_od315_of_diluted_wines", "flavanoids"),  # r=0.79
    "rasio_proline_alkohol": ("proline", "alcohol"),                      # r=0.64
    "rasio_warna_hue": ("color_intensity", "hue"),
}
# Fitur asli yang dibuang pada varian "rasio_ringkas" (redundan dgn rasio)
ASLI_DIBUANG = ["total_phenols", "od280_od315_of_diluted_wines"]
EPS = 1e-6


def muat_data(path="dataset/kasus01_wine.csv"):
    return pd.read_csv(Path(path))


def audit_data(df):
    """Audit: bentuk, tipe, missing, duplikat, nilai tidak wajar, korelasi tinggi."""
    X = df[FITUR_ASLI]
    korelasi = X.corr().abs().where(np.triu(np.ones((13, 13), bool), 1)).stack()
    return {
        "bentuk": df.shape,
        "tipe": df.dtypes.astype(str).to_dict(),
        "missing_total": int(df.isna().sum().sum()),
        "duplikat": int(df.duplicated().sum()),
        "nilai_tak_wajar_<=0": int((X <= 0).sum().sum()),
        "distribusi_kelas": df[TARGET].value_counts().to_dict(),
        "korelasi_tinggi(>0.6)": korelasi[korelasi > 0.6].sort_values(ascending=False).round(3).to_dict(),
    }


class RasioFitur(BaseEstimator, TransformerMixin):
    """Menambah fitur rasio. Tidak punya statistik yang dipelajari (stateless),
    jadi aman dari kebocoran data. Input & output: DataFrame.

    mode:
      'asli'         -> 13 fitur asli saja
      'asli+rasio'   -> 13 asli + 5 rasio
      'rasio_ringkas'-> 5 rasio + asli, tapi total_phenols & od280 dibuang
    """

    def __init__(self, mode="asli+rasio"):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.mode == "asli":
            return X[FITUR_ASLI]
        for nama, (a, b) in RASIO.items():
            X[nama] = X[a] / (X[b] + EPS)
        if self.mode == "rasio_ringkas":
            X = X.drop(columns=ASLI_DIBUANG)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.transform(pd.DataFrame(
            [[1.0] * 13], columns=FITUR_ASLI)).columns)
