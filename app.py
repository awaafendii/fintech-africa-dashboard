import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FinTech Africa — Dashboard Prêts",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #e9ecef;
    }
    .section-title {
        font-size: 12px; font-weight: 600;
        color: #6c757d; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DÉTECTION DU FORMAT CSV
# ─────────────────────────────────────────
def detecter_format(df):
    """Retourne 'nettoye' si le fichier est déjà prétraité, sinon 'brut'."""
    if 'type_compte' in df.columns and 'pays' in df.columns:
        return 'brut'
    return 'nettoye'


def reconstruire_colonnes(df):
    """Reconstruit pays, type_compte et niveau_risque_txt depuis un CSV nettoyé."""

    # Reconstruction de pays
    pays_cols = [c for c in df.columns if c.startswith('pays_')]
    if pays_cols:
        def get_pays(row):
            for c in pays_cols:
                if row[c]:
                    return c.replace('pays_', '')
            return 'Inconnu'
        df['pays'] = df.apply(get_pays, axis=1)
    else:
        df['pays'] = 'Inconnu'

    # Reconstruction de type_compte
    type_cols = [c for c in ['Mobile Money', 'Courant', 'Épargne'] if c in df.columns]
    if type_cols:
        def get_type(row):
            for c in type_cols:
                if row[c] == 1.0 or row[c] == 1:
                    return c
            return 'Inconnu'
        df['type_compte'] = df.apply(get_type, axis=1)
    else:
        df['type_compte'] = 'Inconnu'

    # Reconstruction de niveau_risque texte
    if df['niveau_risque'].dtype in [np.int64, np.float64, int, float]:
        df['niveau_risque_txt'] = df['niveau_risque'].map(
            {0: 'Faible', 1: 'Moyen', 2: 'Élevé', 3: 'Très Élevé'}
        ).fillna('Inconnu')
    else:
        df['niveau_risque_txt'] = df['niveau_risque']

    return df


# ─────────────────────────────────────────
# PIPELINE PRÉTRAITEMENT + MODÈLE
# ─────────────────────────────────────────
@st.cache_data
def charger_et_preparer(chemin_csv: str):
    df = pd.read_csv(chemin_csv)

    # Suppression colonne index parasite
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    format_csv = detecter_format(df)

    # ── CAS 1 : fichier BRUT (original)
    if format_csv == 'brut':
        df = df.drop_duplicates()

        # Outliers IQR
        for col in ['revenu_mensuel', 'montant_transaction']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

        # Imputation
        df['revenu_mensuel']      = df['revenu_mensuel'].fillna(df['revenu_mensuel'].median())
        df['score_credit']        = df['score_credit'].fillna(df['score_credit'].median())
        df['montant_transaction'] = df['montant_transaction'].fillna(df['montant_transaction'].median())
        df['retard_paiement']     = df['retard_paiement'].fillna(df['retard_paiement'].median())
        df['anciennete_client']   = df['anciennete_client'].fillna(df['anciennete_client'].mean())
        df['type_compte']         = df['type_compte'].fillna(df['type_compte'].mode()[0])
        df['pays']                = df['pays'].fillna(df['pays'].mode()[0])
        df['niveau_risque']       = df['niveau_risque'].fillna(df['niveau_risque'].mode()[0])

        # Features
        df['ratio_dette_revenu']  = np.where(df['revenu_mensuel'] > 0, df['montant_transaction'] / df['revenu_mensuel'], 0)
        df['score_risque']        = (df['historique_defaut'] * 2) + df['retard_paiement'] + df['nombre_prets']
        df['revenu_par_annee']    = df['revenu_mensuel'] / (df['anciennete_client'] + 1)
        df['intensite_mobile']    = df['frequence_mobile_money'] / (df['anciennete_client'] + 1)
        df['niveau_risque_enc']   = df['niveau_risque'].map({'Faible': 0, 'Moyen': 1, 'Élevé': 2, 'Très Élevé': 3})
        df['niveau_risque_txt']   = df['niveau_risque']

    # ── CAS 2 : fichier NETTOYÉ (déjà prétraité)
    else:
        df = reconstruire_colonnes(df)

        # Harmonisation des noms de colonnes si nécessaire
        if 'revenu_par_annee_anciennete' in df.columns and 'revenu_par_annee' not in df.columns:
            df['revenu_par_annee'] = df['revenu_par_annee_anciennete']
        if 'intensite_mobile_money' in df.columns and 'intensite_mobile' not in df.columns:
            df['intensite_mobile'] = df['intensite_mobile_money']

        # niveau_risque_enc = la colonne numérique déjà présente
        df['niveau_risque_enc'] = df['niveau_risque']

        # Remplir les éventuelles valeurs manquantes résiduelles
        for col in ['revenu_mensuel', 'score_credit', 'montant_transaction', 'retard_paiement']:
            df[col] = df[col].fillna(df[col].median())
        df['anciennete_client'] = df['anciennete_client'].fillna(df['anciennete_client'].mean())

    # ── MODÈLE RANDOM FOREST (commun aux deux cas)
    FEATURES = [
        'age', 'revenu_mensuel', 'score_credit', 'historique_defaut',
        'montant_transaction', 'frequence_mobile_money', 'retard_paiement',
        'nombre_prets', 'anciennete_client', 'niveau_risque_enc',
        'ratio_dette_revenu', 'score_risque', 'revenu_par_annee', 'intensite_mobile'
    ]

    # Vérification que toutes les features existent
    features_manquantes = [f for f in FEATURES if f not in df.columns]
    if features_manquantes:
        st.error(f"Colonnes manquantes dans le fichier : {features_manquantes}")
        st.stop()

    X = df[FEATURES].astype(float)
    y = df['defaut_paiement'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    df['proba_defaut']         = model.predict_proba(X)[:, 1]
    df['eligible_pret']        = (df['proba_defaut'] < 0.35).astype(int)
    df['montant_pret_suggere'] = np.where(
        df['eligible_pret'] == 1,
        (df['revenu_mensuel'] * 3 * (1 - df['proba_defaut'])).round(-3),
        0
    )

    fi = dict(zip(FEATURES, model.feature_importances_))
    return df, model, FEATURES, accuracy, fi, format_csv


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=64)
st.sidebar.title("FinTech Africa")
st.sidebar.markdown("**Dashboard — Prédiction de prêts**")
st.sidebar.divider()

uploaded = st.sidebar.file_uploader(
    "📂 Charger le dataset CSV",
    type=["csv"],
    help="Accepte le fichier brut OU le fichier déjà nettoyé"
)

if uploaded is None:
    st.info("👈 Chargez **votre fichier** dans la barre latérale gauche.")
    st.stop()

import tempfile, os
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

df, model, FEATURES, accuracy, fi, format_csv = charger_et_preparer(tmp_path)

st.sidebar.divider()
page = st.sidebar.radio("Navigation", ["📊 Vue globale", "👥 Clients", "🤖 Simulation"])
st.sidebar.divider()
badge = "✅ Fichier nettoyé détecté" if format_csv == 'nettoye' else "📄 Fichier brut détecté"
st.sidebar.caption(badge)
st.sidebar.caption(f"**{len(df)}** clients  |  Précision : **{accuracy*100:.1f}%**")


# ═══════════════════════════════════════════════════════
# PAGE 1 — VUE GLOBALE
# ═══════════════════════════════════════════════════════
if page == "📊 Vue globale":
    st.title("📊 Vue globale — FinTech Africa")
    st.markdown("Analyse comportementale et prédiction d'éligibilité aux prêts bancaires.")
    st.divider()

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Total clients",     len(df))
    col2.metric("✅ Éligibles",      int(df['eligible_pret'].sum()),
                f"{df['eligible_pret'].mean()*100:.1f}%")
    col3.metric("❌ Non éligibles",  int((df['eligible_pret']==0).sum()),
                f"{(df['eligible_pret']==0).mean()*100:.1f}%")
    col4.metric("Taux défaut réel",  f"{df['defaut_paiement'].mean()*100:.1f}%")
    col5.metric("Revenu moyen",      f"{df['revenu_mensuel'].mean()/1000:.0f}k F")
    col6.metric("Score crédit moy",  f"{df['score_credit'].mean():.0f}")
    col7.metric("Précision modèle",  f"{accuracy*100:.1f}%")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Éligibilité par pays**")
        pays_stats = df.groupby('pays').agg(
            Éligibles=('eligible_pret', 'sum'),
            Total=('eligible_pret', 'count')
        ).reset_index()
        pays_stats['Non éligibles'] = pays_stats['Total'] - pays_stats['Éligibles']
        fig = px.bar(pays_stats, x='pays', y=['Éligibles', 'Non éligibles'],
                     color_discrete_map={'Éligibles': '#639922', 'Non éligibles': '#E24B4A'},
                     barmode='stack', height=300)
        fig.update_layout(margin=dict(t=10, b=10), legend_title_text='',
                          xaxis_title='', yaxis_title='Clients')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Éligibilité par niveau de risque**")
        risque_ordre = ['Faible', 'Moyen', 'Élevé', 'Très Élevé']
        risque_stats = df.groupby('niveau_risque_txt').agg(
            Éligibles=('eligible_pret', 'sum'),
            Total=('eligible_pret', 'count')
        ).reindex(risque_ordre).reset_index()
        risque_stats['Non éligibles'] = risque_stats['Total'] - risque_stats['Éligibles']
        fig2 = px.bar(risque_stats, x='niveau_risque_txt', y=['Éligibles', 'Non éligibles'],
                      color_discrete_map={'Éligibles': '#639922', 'Non éligibles': '#E24B4A'},
                      barmode='stack', height=300)
        fig2.update_layout(margin=dict(t=10, b=10), legend_title_text='',
                           xaxis_title='', yaxis_title='Clients')
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Importance des variables (Random Forest)**")
        fi_df = pd.DataFrame({
            'Variable': list(fi.keys()),
            'Importance (%)': [round(v * 100, 1) for v in fi.values()]
        }).sort_values('Importance (%)')
        fig3 = px.bar(fi_df, x='Importance (%)', y='Variable', orientation='h',
                      height=420, color='Importance (%)',
                      color_continuous_scale='Blues')
        fig3.update_layout(margin=dict(t=10, b=10),
                           coloraxis_showscale=False, yaxis_title='')
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown("**Distribution des probabilités de défaut**")
        fig4 = px.histogram(df, x='proba_defaut', nbins=20,
                            color_discrete_sequence=['#378ADD'], height=420,
                            labels={'proba_defaut': 'Probabilité de défaut'})
        fig4.add_vline(x=0.35, line_dash="dash", line_color="#E24B4A",
                       annotation_text="Seuil éligibilité (35%)")
        fig4.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.markdown("**Taux d'éligibilité et revenu moyen par pays**")
    pays_detail = df.groupby('pays').agg(
        taux_elig=('eligible_pret',  lambda x: round(x.mean() * 100, 1)),
        revenu_moy=('revenu_mensuel', lambda x: round(x.mean() / 1000, 0)),
        nb_clients=('client_id',     'count'),
        score_moy=('score_credit',   lambda x: round(x.mean(), 0))
    ).reset_index()
    pays_detail.columns = ['Pays', 'Taux éligibilité (%)', 'Revenu moyen (k FCFA)', 'Nb clients', 'Score crédit moy']
    st.dataframe(pays_detail, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# PAGE 2 — CLIENTS
# ═══════════════════════════════════════════════════════
elif page == "👥 Clients":
    st.title("👥 Liste des clients")
    st.divider()

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        filtre_elig   = st.selectbox("Éligibilité", ["Tous", "✅ Éligibles", "❌ Non éligibles"])
    with f2:
        filtre_pays   = st.selectbox("Pays", ["Tous"] + sorted(df['pays'].unique().tolist()))
    with f3:
        filtre_risque = st.selectbox("Niveau de risque", ["Tous", "Faible", "Moyen", "Élevé", "Très Élevé"])
    with f4:
        search = st.text_input("🔍 Rechercher un client (ID)", "")

    dff = df.copy()
    if filtre_elig   == "✅ Éligibles":    dff = dff[dff['eligible_pret'] == 1]
    elif filtre_elig == "❌ Non éligibles": dff = dff[dff['eligible_pret'] == 0]
    if filtre_pays   != "Tous":            dff = dff[dff['pays'] == filtre_pays]
    if filtre_risque != "Tous":            dff = dff[dff['niveau_risque_txt'] == filtre_risque]
    if search:                             dff = dff[dff['client_id'].str.contains(search.upper())]

    st.caption(f"**{len(dff)}** clients affichés")

    affichage = dff[['client_id', 'age', 'revenu_mensuel', 'score_credit',
                     'pays', 'type_compte', 'niveau_risque_txt',
                     'proba_defaut', 'eligible_pret', 'montant_pret_suggere']].copy()
    affichage.columns = ['Client', 'Âge', 'Revenu (FCFA)', 'Score crédit',
                         'Pays', 'Type compte', 'Niveau risque',
                         'Prob. défaut', 'Éligible', 'Prêt suggéré (FCFA)']
    affichage['Prob. défaut']        = affichage['Prob. défaut'].apply(lambda x: f"{x*100:.1f}%")
    affichage['Éligible']            = affichage['Éligible'].apply(lambda x: "✅ Oui" if x else "❌ Non")
    affichage['Revenu (FCFA)']       = affichage['Revenu (FCFA)'].apply(lambda x: f"{int(x):,}".replace(',', ' '))
    affichage['Prêt suggéré (FCFA)'] = affichage['Prêt suggéré (FCFA)'].apply(
        lambda x: f"{int(x):,}".replace(',', ' ') if x > 0 else "—")

    st.dataframe(affichage, use_container_width=True, hide_index=True, height=500)

    st.divider()
    st.markdown("### 🔎 Fiche détaillée d'un client")
    client_ids = sorted(dff['client_id'].tolist())
    if client_ids:
        client_sel = st.selectbox("Sélectionner un client", client_ids)
        c = dff[dff['client_id'] == client_sel].iloc[0]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"**Client :** {c['client_id']}")
            st.markdown(f"**Âge :** {int(c['age'])} ans")
            st.markdown(f"**Pays :** {c['pays']}")
            st.markdown(f"**Type de compte :** {c['type_compte']}")
            st.markdown(f"**Niveau de risque :** {c['niveau_risque_txt']}")
        with col_b:
            st.markdown(f"**Revenu mensuel :** {int(c['revenu_mensuel']):,} FCFA".replace(',', ' '))
            st.markdown(f"**Score de crédit :** {int(c['score_credit'])}")
            st.markdown(f"**Historique défaut :** {int(c['historique_defaut'])}")
            st.markdown(f"**Retard paiement :** {int(c['retard_paiement'])} jours")
            st.markdown(f"**Nombre de prêts :** {int(c['nombre_prets'])}")
            st.markdown(f"**Ancienneté :** {int(c['anciennete_client'])} ans")
            st.markdown(f"**Mobile Money :** {int(c['frequence_mobile_money'])} tx/mois")
        with col_c:
            prob = c['proba_defaut']
            st.metric("Probabilité de défaut", f"{prob*100:.1f}%")
            if c['eligible_pret']:
                st.success(f"✅ Éligible — Prêt suggéré : **{int(c['montant_pret_suggere']):,} FCFA**".replace(',', ' '))
            else:
                st.error("❌ Non éligible au prêt")
            st.markdown("**Features calculées :**")
            st.markdown(f"- ratio_dette_revenu : `{c['ratio_dette_revenu']:.3f}`")
            st.markdown(f"- score_risque : `{c['score_risque']:.1f}`")
            st.markdown(f"- revenu_par_annee : `{c['revenu_par_annee']:,.0f}`")
            st.markdown(f"- intensite_mobile : `{c['intensite_mobile']:.2f}`")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            title={'text': "Probabilité de défaut (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E24B4A" if prob > 0.5 else "#EF9F27" if prob > 0.35 else "#639922"},
                'steps': [
                    {'range': [0,  35], 'color': '#EAF3DE'},
                    {'range': [35, 60], 'color': '#FFF3CD'},
                    {'range': [60,100], 'color': '#FCEBEB'},
                ],
                'threshold': {'line': {'color': "#E24B4A", 'width': 3}, 'value': 35}
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.warning("Aucun client ne correspond aux filtres sélectionnés.")


# ═══════════════════════════════════════════════════════
# PAGE 3 — SIMULATION
# ═══════════════════════════════════════════════════════
elif page == "🤖 Simulation":
    st.title("🤖 Simulation — Nouveau client")
    st.markdown("Renseignez le profil d'un nouveau client pour prédire son éligibilité au prêt.")
    st.divider()

    col_form, col_res = st.columns([1, 1])

    with col_form:
        st.markdown("#### Profil client")
        age    = st.slider("Âge", 18, 75, 38)
        revenu = st.slider("Revenu mensuel (FCFA)", 50_000, 800_000, 300_000, step=10_000)
        score  = st.slider("Score de crédit", 200, 900, 600)
        mt     = st.slider("Montant transaction (FCFA)", 3_000, 200_000, 50_000, step=1_000)
        hd     = st.slider("Historique défaut", 0, 5, 0)
        retard = st.slider("Retard paiement (jours)", 0, 30, 0)
        np_    = st.slider("Nombre de prêts", 0, 15, 2)
        anc    = st.slider("Ancienneté client (ans)", 1, 30, 5)
        mm     = st.slider("Fréquence Mobile Money (tx/mois)", 0, 50, 10)
        risque_lib = st.selectbox("Niveau de risque", ['Faible', 'Moyen', 'Élevé', 'Très Élevé'])
        risque_enc = {'Faible': 0, 'Moyen': 1, 'Élevé': 2, 'Très Élevé': 3}[risque_lib]

    with col_res:
        st.markdown("#### Résultat de prédiction")

        ratio   = mt / revenu if revenu > 0 else 0
        scoreR  = hd * 2 + retard + np_
        rev_anc = revenu / (anc + 1)
        int_mob = mm / (anc + 1)

        X_sim = pd.DataFrame([[
            age, revenu, score, hd, mt, mm, retard,
            np_, anc, risque_enc, ratio, scoreR, rev_anc, int_mob
        ]], columns=FEATURES)

        proba    = model.predict_proba(X_sim)[0][1]
        eligible = proba < 0.35
        pret_sugg = int(revenu * 3 * (1 - proba) / 1000) * 1000 if eligible else 0

        if eligible:
            st.success("### ✅ Client éligible au prêt")
            st.metric("Probabilité de défaut", f"{proba*100:.1f}%",
                      delta=f"-{(0.35-proba)*100:.1f}% sous le seuil")
            st.metric("Montant de prêt suggéré", f"{pret_sugg:,} FCFA".replace(',', ' '))
        else:
            st.error("### ❌ Client non éligible")
            st.metric("Probabilité de défaut", f"{proba*100:.1f}%",
                      delta=f"+{(proba-0.35)*100:.1f}% au-dessus du seuil",
                      delta_color="inverse")

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(proba * 100, 1),
            delta={'reference': 35,
                   'increasing': {'color': '#E24B4A'},
                   'decreasing': {'color': '#639922'}},
            title={'text': "Probabilité de défaut (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#E24B4A" if proba > 0.5 else "#EF9F27" if proba > 0.35 else "#639922"},
                'steps': [
                    {'range': [0,  35], 'color': '#EAF3DE'},
                    {'range': [35, 60], 'color': '#FFF3CD'},
                    {'range': [60,100], 'color': '#FCEBEB'},
                ],
                'threshold': {'line': {'color': "#E24B4A", 'width': 4}, 'value': 35}
            }
        ))
        fig_g.update_layout(height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        st.markdown("#### Features calculées automatiquement")
        feat_df = pd.DataFrame({
            'Feature':     ['ratio_dette_revenu', 'score_risque', 'revenu_par_annee', 'intensite_mobile'],
            'Valeur':      [f"{ratio:.3f}", f"{scoreR}", f"{rev_anc:,.0f}", f"{int_mob:.2f}"],
            'Description': ['Transaction / Revenu', 'Défaut×2 + Retard + Nb prêts',
                            'Revenu / (Ancienneté + 1)', 'Mobile Money / (Ancienneté + 1)']
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        st.markdown("#### Comparaison avec la moyenne du portefeuille")
        comp_df = pd.DataFrame({
            'Variable':             ['Revenu mensuel', 'Score crédit', 'Prob. défaut'],
            'Ce client':            [f"{revenu:,} FCFA".replace(',', ' '), str(score), f"{proba*100:.1f}%"],
            'Moyenne portefeuille': [
                f"{int(df['revenu_mensuel'].mean()):,} FCFA".replace(',', ' '),
                f"{df['score_credit'].mean():.0f}",
                f"{df['proba_defaut'].mean()*100:.1f}%"
            ]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
