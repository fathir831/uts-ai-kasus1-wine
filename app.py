"""app.py - aplikasi Streamlit klasifikasi kultivar anggur (Kasus 1)."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import fitur  # noqa: F401  (diperlukan joblib untuk memuat RasioFitur)

PATH_MODEL = Path("models") / "wine_pipeline.joblib"
LABEL = {
    "alcohol": "Alkohol", "malic_acid": "Asam malat", "ash": "Abu",
    "alcalinity_of_ash": "Alkalinitas abu", "magnesium": "Magnesium",
    "total_phenols": "Total fenol", "flavanoids": "Flavanoid",
    "nonflavanoid_phenols": "Fenol non-flavanoid",
    "proanthocyanins": "Proantosianin", "color_intensity": "Intensitas warna",
    "hue": "Hue", "od280_od315_of_diluted_wines": "OD280/OD315",
    "proline": "Prolin",
}


@st.cache_resource
def muat_model():
    bundel = joblib.load(PATH_MODEL)
    return bundel["pipeline"], bundel["metadata"]


def prediksi(pipe, meta, nilai: dict):
    """Input mentah (13 fitur asli) -> (label, tabel probabilitas)."""
    X = pd.DataFrame([nilai], columns=meta["fitur_input"])
    proba = pipe.predict_proba(X)[0]
    tabel = pd.Series(proba, index=pipe.classes_, name="Probabilitas")
    return tabel.idxmax(), tabel


def main():
    st.set_page_config(page_title="Klasifikasi Kultivar Anggur", page_icon="🍷")
    st.title("🍷 Klasifikasi Kultivar Anggur")
    st.caption("Input hasil analisis kimia mentah; scaling dan fitur rasio diproses otomatis oleh pipeline.")

    if not PATH_MODEL.exists():
        st.error("Model belum ada. Jalankan `python latih.py` terlebih dahulu.")
        st.stop()
    pipe, meta = muat_model()

    tab_pred, tab_metrik = st.tabs(["Prediksi", "Metrik evaluasi"])

    with tab_pred:
        st.subheader("Masukkan hasil analisis kimia")
        nilai = {}
        kiri, kanan = st.columns(2)
        for i, kol in enumerate(meta["fitur_input"]):
            lo, hi, med = meta["rentang_fitur"][kol]
            margin = (hi - lo) * 0.25
            with (kiri if i % 2 == 0 else kanan):
                nilai[kol] = st.number_input(
                    LABEL.get(kol, kol), min_value=max(0.0, lo - margin),
                    max_value=hi + margin, value=med, format="%.3f",
                    help=f"Rentang data latih: {lo:.2f} - {hi:.2f}")

        if st.button("Prediksi", type="primary"):
            label, tabel = prediksi(pipe, meta, nilai)
            st.success(f"Prediksi: **{label}** ({tabel.max():.1%})")
            st.bar_chart(tabel)
            st.dataframe(tabel.map("{:.2%}".format).to_frame())
            diluar = [k for k, v in nilai.items()
                      if not meta["rentang_fitur"][k][0] <= v <= meta["rentang_fitur"][k][1]]
            if diluar:
                st.warning("Nilai di luar rentang data latih (prediksi kurang andal): "
                           + ", ".join(diluar))

    with tab_metrik:
        m = meta["metrik_test"]
        st.write(f"Model: **{meta['model_terpilih']}** | varian fitur: `{meta['mode_fitur']}` "
                 f"| train={meta['n_train']}, test={meta['n_test']}")
        c = st.columns(4)
        c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
        c[1].metric("F1 macro", f"{m['f1_macro']:.3f}")
        c[2].metric("Precision macro", f"{m['precision_macro']:.3f}")
        c[3].metric("Recall macro", f"{m['recall_macro']:.3f}")
        st.subheader("Confusion matrix (data test)")
        st.dataframe(pd.DataFrame(meta["confusion_matrix"], index=[f"Aktual {k}" for k in meta["kelas"]],
                                  columns=[f"Prediksi {k}" for k in meta["kelas"]]))
        st.subheader("Classification report")
        rep = pd.DataFrame(meta["classification_report"]).T.round(3)
        st.dataframe(rep)
        st.subheader("Perbandingan model (cross-validation di data train)")
        st.dataframe(pd.DataFrame(meta["hasil_cv"]).round(4))


main()
