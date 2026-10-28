from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Auto MPG", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "mpg.xlsx"
FEATURES = [
    "cylinders", "displacement", "horsepower",
    "weight_kg", "acceleration", "model_year", "origin",
]
NUMERIC_COLS = ["cylinders", "mpg", "horsepower", "weight", "displacement", "acceleration"]
CYLINDER_OPTIONS = [3, 4, 5, 6, 8]
COLORS = {"usa": "#0068C9", "japan": "#83C9FF", "europe": "#FF2B2B"}


def mpg_to_l100(mpg):
    """Convertit des miles par gallon (US) en L/100 km."""
    return 235.215 / mpg


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    # Noms de colonnes homogènes
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.rename(columns={"year": "model_year", "car_name": "name"})
    # Origine : gère les codes 1/2/3 ou le texte avec majuscules
    df["origin"] = (
        df["origin"].replace({1: "usa", 2: "europe", 3: "japan"})
        .astype(str).str.strip().str.lower()
    )
    # Valeurs non numériques (ex. "?") -> NaN
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cylinders"] = pd.to_numeric(df["cylinders"], errors="coerce")
    if "name" not in df.columns:
        df["name"] = ""
    df = df.dropna(subset=NUMERIC_COLS)
    df["weight_kg"] = (df["weight"] * 0.4536).round(0)  # lbs -> kg
    df["l_100km"] = mpg_to_l100(df["mpg"])
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    X, y = df[FEATURES], df["mpg"]
    pre = ColumnTransformer(
        [("origin", OneHotEncoder(handle_unknown="ignore"), ["origin"])],
        remainder="passthrough",
    )
    pipe = Pipeline(
        [("pre", pre), ("rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))]
    )
    # On prédit log(mpg) : la relation avec le poids/la puissance est plus linéaire
    model = TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)

    # Validation croisée 5 plis : plus fiable qu'un seul découpage train/test
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
    mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()

    model.fit(X, y)
    fitted = model.regressor_
    importances = pd.Series(
        fitted.named_steps["rf"].feature_importances_,
        index=[n.split("__")[-1] for n in fitted.named_steps["pre"].get_feature_names_out()],
    ).sort_values()
    return model, r2, mae, importances


df = load_data()

# ---------------------------------------------------------------- Sidebar
st.sidebar.header("Filtres")
origin = st.sidebar.selectbox("Origine", ["Toutes"] + sorted(df["origin"].unique()))
cylinders = st.sidebar.multiselect(
    "Nombre de cylindres",
    CYLINDER_OPTIONS,
    default=CYLINDER_OPTIONS,
)
y_min, y_max = int(df["model_year"].min()), int(df["model_year"].max())
years = st.sidebar.slider("Année du modèle", y_min, y_max, (y_min, y_max))
mpg_min = st.sidebar.slider(
    "Efficacité minimale (mpg)",
    float(df["mpg"].min()), float(df["mpg"].max()), float(df["mpg"].min()),
)

mask = (
    df["cylinders"].isin(cylinders)
    & df["model_year"].between(*years)
    & (df["mpg"] >= mpg_min)
)
if origin != "Toutes":
    mask &= df["origin"] == origin
fdf = df[mask]

st.title("Consommation des véhicules")
tab_desc, tab_ml = st.tabs(["Analyse descriptive", "Prédiction (Random Forest)"])

# ------------------------------------------------- Analyse descriptive
with tab_desc:
    if fdf.empty:
        st.warning("Aucun véhicule ne correspond aux filtres.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Véhicules", len(fdf))
        c2.metric("Puissance moyenne", f"{fdf['horsepower'].mean():.0f} ch")
        c3.metric("Efficacité moyenne", f"{fdf['mpg'].mean():.1f} mpg")
        c4.metric("Consommation moyenne", f"{fdf['l_100km'].mean():.1f} L/100 km")

        with st.expander("Voir les données filtrées"):
            st.dataframe(fdf, width="stretch")

        fig = px.scatter(
            fdf, x="weight_kg", y="mpg", color="origin",
            color_discrete_map=COLORS, hover_name="name",
            labels={"weight_kg": "Poids (kg)", "mpg": "Efficacité (mpg)", "origin": "Origine"},
            title="Poids et efficacité énergétique",
        )
        st.plotly_chart(fig, width="stretch")
        corr = fdf["weight_kg"].corr(fdf["mpg"])
        st.caption(f"Corrélation poids / efficacité : **{corr:.2f}**")

        by_year = fdf.groupby("model_year", as_index=False)["mpg"].mean()
        fig = px.line(
            by_year, x="model_year", y="mpg", markers=True,
            labels={"model_year": "Année du modèle", "mpg": "Efficacité moyenne (mpg)"},
            title="Efficacité moyenne par année de modèle",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Comparaison par origine")
        by_origin = fdf.groupby("origin").agg(
            vehicules=("mpg", "size"),
            mpg_moyen=("mpg", "mean"),
            l_100km_moyen=("l_100km", "mean"),
            puissance_moyenne=("horsepower", "mean"),
            poids_moyen_kg=("weight_kg", "mean"),
        ).round(1)
        st.dataframe(by_origin, width="stretch")

        col_a, col_b = st.columns(2)
        fig = px.box(
            fdf, x="origin", y="mpg", color="origin", color_discrete_map=COLORS,
            labels={"origin": "Origine", "mpg": "Efficacité (mpg)"},
            title="Distribution de l'efficacité par origine",
        )
        col_a.plotly_chart(fig, width="stretch")
        fig = px.bar(
            by_origin.reset_index(), x="origin", y="puissance_moyenne",
            color="origin", color_discrete_map=COLORS,
            labels={"origin": "Origine", "puissance_moyenne": "Puissance moyenne (ch)"},
            title="Puissance moyenne par origine",
        )
        col_b.plotly_chart(fig, width="stretch")

# ------------------------------------------------------- Prédiction ML
with tab_ml:
    model, r2, mae, importances = train_model(df)
    m1, m2 = st.columns(2)
    m1.metric("R² (validation croisée 5 plis)", f"{r2:.2f}")
    m2.metric("Erreur absolue moyenne (validation croisée)", f"{mae:.2f} mpg")

    st.subheader("Caractéristiques du véhicule")
    with st.form("prediction"):
        a, b = st.columns(2)
        cyl = a.selectbox("Cylindres", CYLINDER_OPTIONS, index=1)
        org = a.selectbox("Origine", sorted(df["origin"].unique()))
        year = a.slider("Année du modèle", y_min, y_max, y_max)
        disp = b.number_input(
            "Cylindrée (cu in)", float(df["displacement"].min()),
            float(df["displacement"].max()), float(df["displacement"].median()),
        )
        hp = b.number_input(
            "Puissance (ch)", float(df["horsepower"].min()),
            float(df["horsepower"].max()), float(df["horsepower"].median()),
        )
        wt = b.number_input(
            "Poids (kg)", float(df["weight_kg"].min()),
            float(df["weight_kg"].max()), float(df["weight_kg"].median()),
        )
        acc = a.slider(
            "Accélération (0-60 mph, s)", float(df["acceleration"].min()),
            float(df["acceleration"].max()), float(df["acceleration"].median()),
        )
        submitted = st.form_submit_button("Estimer la consommation")

    if submitted:
        x_new = pd.DataFrame([{
            "cylinders": cyl, "displacement": disp, "horsepower": hp,
            "weight_kg": wt, "acceleration": acc, "model_year": year, "origin": org,
        }])
        pred = float(model.predict(x_new)[0])
        r1, r2_ = st.columns(2)
        r1.metric("Efficacité estimée", f"{pred:.1f} mpg")
        r2_.metric("Consommation estimée", f"{mpg_to_l100(pred):.1f} L/100 km")

    st.subheader("Importance des variables")
    fig = px.bar(importances, orientation="h", labels={"value": "Importance", "index": ""})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")