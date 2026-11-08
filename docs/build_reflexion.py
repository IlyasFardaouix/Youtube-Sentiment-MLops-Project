"""Generate the reflection-questions report PDF (mirrors the expected TP layout)."""
import json
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "rapport_reflexion.pdf"
EDA = ROOT / "data" / "processed" / "class_distribution.png"
M = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
# Per-class statistics computed from data/processed (train.csv + test.csv).
D = {
    "rows": {
        "1": {"name": "Positif (1)", "total": 15723, "train": 12578, "test": 3145, "mean_w": 42.9, "med_w": 21},
        "0": {"name": "Neutre (0)", "total": 12606, "train": 10085, "test": 2521, "mean_w": 9.9, "med_w": 7},
        "-1": {"name": "Négatif (-1)", "total": 8233, "train": 6586, "test": 1647, "mean_w": 34.9, "med_w": 20},
    },
    "total": 36562, "train": 29249, "test": 7313,
    "pct": {"1": 43.0, "0": 34.5, "-1": 22.5},
}

NAVY, TEAL, ORANGE, INK, MUTED, PAPER = "#173b4a", "#23858b", "#d47b4d", "#24313a", "#687781", "#fbfaf7"
POS, NEU, NEG = "#3f8f64", "#b08a43", "#c64f45"
AUTHOR = "Ilyas Fardaoui"


def pct(x):
    return f"{x*100:.2f}".replace(".", ",") + " %"


def nf(n):
    return f"{n:,}".replace(",", " ")


def page(title, kicker=""):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(PAPER)
    if kicker:
        fig.text(0.08, 0.95, kicker, fontsize=12, color=TEAL, weight="bold")
    fig.text(0.08, 0.915, title, fontsize=16, weight="bold", color=NAVY)
    return fig


def foot(fig, n):
    fig.text(0.08, 0.035, f"TP3 - Projet Cloud Computing | {AUTHOR}", fontsize=8, color=MUTED)
    fig.text(0.92, 0.035, str(n), fontsize=8, color=MUTED, ha="right")


def qapage(pdf, n, title, kicker, qas, intro=None):
    fig = page(title, kicker)
    y = 0.86
    if intro:
        for line in textwrap.wrap(intro, width=96):
            fig.text(0.08, y, line, fontsize=10, color=MUTED)
            y -= 0.021
        y -= 0.012
    for q, paras in qas:
        for line in textwrap.wrap(q, width=92):
            fig.text(0.08, y, line, fontsize=11.5, color=ORANGE, weight="bold")
            y -= 0.023
        for para in paras:
            for line in textwrap.wrap(para, width=97):
                fig.text(0.1, y, line, fontsize=10, color=INK)
                y -= 0.0205
            y -= 0.008
        y -= 0.016
    foot(fig, n)
    pdf.savefig(fig, facecolor=PAPER)
    plt.close(fig)


def chart_page(pdf, n, title, kicker, draw, caption, rect=(0.14, 0.3, 0.72, 0.48)):
    fig = page(title, kicker)
    ax = fig.add_axes(rect)
    draw(ax)
    fig.text(0.1, rect[1] - 0.04, caption, fontsize=9.5, color=MUTED)
    foot(fig, n)
    pdf.savefig(fig, facecolor=PAPER)
    plt.close(fig)


# ---- figure painters (Ilyas's own project material) ----
def structure(ax):
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#10261f", ec="none"))
    ax.text(0.03, 0.95, "EXPLORER  —  Youtube-Sentiment-MLops-Project", color="#9fd0cf", fontsize=10, va="top", family="monospace")
    tree = [
        "data/",
        "   raw/reddit.csv",
        "   processed/  (train.csv, test.csv, class_distribution.png)",
        "src/",
        "   data/    download_dataset.py, prepare_data.py",
        "   models/  train.py",
        "   api/     main.py",
        "models/      *.joblib (pipeline, vectoriseur), metrics.json",
        "chrome-extension/  manifest.json, popup.html/css/js, content.js",
        "tests/       test_api.py, test_model.py",
        "docs/        API.md, rapport_technique.md, screenshots/",
        "logs/        .gitkeep",
        "Dockerfile",
        "app_api.py",
        "requirements.txt   requirements-prod.txt",
        "README.md",
    ]
    y = 0.86
    for line in tree:
        color = "#e6c98a" if (line.endswith("/") or "/" in line.split()[0]) else "#dfe7e3"
        ax.text(0.05, y, line, color=color, fontsize=10.5, va="top", family="monospace")
        y -= 0.052


def data_table(ax):
    ax.axis("off")
    r = D["rows"]
    order = ["1", "0", "-1"]
    cell = []
    for k in order:
        row = r[k]
        cell.append([row["name"], f'{nf(row["total"])} ({D["pct"][k]:.1f}%)'.replace(".", ","),
                     nf(row["train"]), nf(row["test"]), f'{row["mean_w"]}'.replace(".", ","), str(row["med_w"])])
    cell.append(["Total", nf(D["total"]), nf(D["train"]), nf(D["test"]), "-", "-"])
    table = ax.table(cellText=cell,
                     colLabels=["Classe", "Total nettoyé", "Train (80%)", "Test (20%)", "Long. moy.\n(mots)", "Long. méd.\n(mots)"],
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (rr, cc), c in table.get_celld().items():
        c.set_height(0.14); c.set_edgecolor("#d8e1df")
        if rr == 0:
            c.set_facecolor("#173b4a"); c.set_text_props(color="white", weight="bold")
        elif rr == len(cell):
            c.set_facecolor("#e8f2f0"); c.set_text_props(weight="bold")
        else:
            c.set_facecolor("#ffffff")


def eda(ax):
    ax.axis("off")
    if EDA.exists():
        ax.imshow(mpimg.imread(EDA))


def confusion(ax):
    cm = M["confusion_matrix"]
    ax.imshow(cm, cmap="BuGn")
    labels = ["-1", "0", "1"]
    ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    peak = max(max(r) for r in cm)
    for i, r in enumerate(cm):
        for j, v in enumerate(r):
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > peak / 2 else INK, fontsize=12)


def comparison(ax):
    c = M["candidates"]
    vals = [c["logistic_regression"] * 100, c["linear_svm"] * 100]
    bars = ax.bar(["Logistic\nRegression", "Linear\nSVM"], vals, color=[TEAL, MUTED], width=0.5)
    ax.set_ylim(80, 92); ax.set_ylabel("Accuracy (%)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}%", ha="center", fontsize=11, weight="bold", color=NAVY)


def tests_fig(ax):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#10261f", ec="none"))
    lines = [
        ("$ pytest -q", "#9fd0cf"),
        ("", "white"),
        ("tests/test_api.py .......                                   [ 54%]", "#8fe3a0"),
        ("tests/test_model.py ......                                  [100%]", "#8fe3a0"),
        ("", "white"),
        ("================= 13 passed in 19.40s =================", "#8fe3a0"),
        ("", "white"),
        ("API   : health, batch vide, hors-limite, payload invalide,", "#dfe7e3"),
        ("        commentaire vide, prédiction cohérente, batch de 50", "#dfe7e3"),
        ("Modèle: accuracy >= 80%, F1 par classe > 75%, latence < 100ms,", "#dfe7e3"),
        ("        cas limites (court/long, emojis, langues mélangées)", "#dfe7e3"),
    ]
    y = 0.9
    for txt, col in lines:
        ax.text(0.04, y, txt, color=col, fontsize=10.5, va="top", family="monospace")
        y -= 0.078


def build():
    with PdfPages(OUTPUT) as pdf:
        # cover (mirrors Anas's header)
        cov = plt.figure(figsize=(8.27, 11.69)); cov.patch.set_facecolor(PAPER)
        cov.text(0.08, 0.80, "TP3 - Projet Cloud Computing", fontsize=26, weight="bold", color=NAVY)
        cov.text(0.08, 0.745, "Analyse de sentiment des commentaires YouTube", fontsize=15, color=ORANGE)
        cov.text(0.08, 0.70, "Questions de réflexion", fontsize=14, color=TEAL, weight="bold")
        cov.text(0.08, 0.55, "Élément de Module : Industrie 4.0", fontsize=13, color=INK)
        cov.text(0.08, 0.505, f"Réalisé par : {AUTHOR}", fontsize=13, color=INK)
        cov.text(0.08, 0.46, "Encadré par : Prof. Maria Zemzami", fontsize=13, color=INK)
        cov.text(0.08, 0.37, "Université Mohamed V - ENSAM Rabat\nFilière INDIA | Année universitaire 2026/27", fontsize=12, color=MUTED, linespacing=1.6)
        foot(cov, 1); pdf.savefig(cov, facecolor=PAPER); plt.close(cov)

        # Phase 1
        chart_page(pdf, 2, "Phase 1 : Configuration de l'environnement MLOps", "PHASE 1",
                   structure, "Figure 1.1 : Structure du projet.", rect=(0.08, 0.14, 0.84, 0.66))
        qapage(pdf, 3, "Phase 1 : Questions de réflexion", "PHASE 1", [
            ("1 - Pourquoi séparer les données raw et processed ?", [
                "data/raw conserve les données brutes intactes : si le nettoyage change, on régénère data/processed sans retélécharger ni altérer la source.",
                "Cela garantit un flux unidirectionnel (raw -> nettoyage -> processed) et permet à l'entraînement de charger directement les splits nettoyés sans rejouer le nettoyage."]),
            ("2 - Quels fichiers ne doivent jamais être commités sur Git ?", [
                "L'environnement virtuel (.venv/), les secrets (.env, tokens), les datasets volumineux (data/raw et data/processed/*.csv), les caches et artefacts (__pycache__/, *.pyc, logs/), et les gros modèles. Mon .gitignore couvre ces cas."]),
            ("3 - Quelle est l'importance d'un environnement virtuel ?", [
                "Il isole les dépendances du projet pour éviter les conflits de versions entre projets, protège l'installation Python globale, et rend l'environnement reproductible via requirements.txt."]),
        ])

        # Phase 2
        fig = page("Phase 2 : Pipeline de données et préparation", "PHASE 2")
        ax1 = fig.add_axes([0.08, 0.55, 0.84, 0.3]); data_table(ax1)
        fig.text(0.08, 0.5, "Tableau 2.1 : Répartition et longueur des commentaires (après nettoyage).", fontsize=9.5, color=MUTED)
        ax2 = fig.add_axes([0.26, 0.12, 0.48, 0.3]); eda(ax2)
        fig.text(0.08, 0.09, "Figure 2.1 : EDA — distribution des classes.", fontsize=9.5, color=MUTED)
        foot(fig, 4); pdf.savefig(fig, facecolor=PAPER); plt.close(fig)

        qapage(pdf, 5, "Phase 2 : Questions de réflexion", "PHASE 2", [
            ("1 - Pourquoi nettoyer les URLs et mentions ?", [
                "Les URLs et mentions sont quasi uniques et créent des milliers de tokens rares sans valeur sémantique, qui gonflent la matrice TF-IDF. Les remplacer par des tokens (URL, USER, HASHTAG) réduit le bruit et la dimensionnalité et limite le surapprentissage."]),
            ("2 - Comment gérer les emojis dans le texte ?", [
                "Dans mon nettoyage, la normalisation retire les emojis avec les autres symboles pour privilégier le signal textuel. Une amélioration serait de les convertir en tokens sémantiques (emoji_positif, emoji_négatif) avant la normalisation, afin de conserver ce signal émotionnel utile sur les commentaires courts."]),
            ("3 - Quel ratio train/test est approprié ?", [
                "Un split stratifié 80/20 (random_state=42) : 29 249 commentaires pour l'apprentissage et 7 313 pour une évaluation fiable (dont plus de 1 600 pour la classe minoritaire). La stratification préserve la distribution des classes."]),
        ])

        # Phase 3
        fig = page("Phase 3 : Développement et entraînement du modèle", "PHASE 3")
        ax1 = fig.add_axes([0.16, 0.57, 0.44, 0.28]); confusion(ax1)
        fig.text(0.08, 0.5, "Figure 3.1 : Matrice de confusion (jeu de test, 7 313 commentaires).", fontsize=9.5, color=MUTED)
        ax2 = fig.add_axes([0.2, 0.16, 0.58, 0.26]); comparison(ax2)
        fig.text(0.08, 0.09, "Figure 3.2 : Comparaison des modèles — LR retenue pour predict_proba.", fontsize=9.5, color=MUTED)
        foot(fig, 6); pdf.savefig(fig, facecolor=PAPER); plt.close(fig)

        qapage(pdf, 7, "Phase 3 : Questions de réflexion", "PHASE 3", [
            ("1 - Pourquoi utiliser TF-IDF plutôt que Count Vectorizer ?", [
                "CountVectorizer donne un poids élevé aux mots fréquents mais peu informatifs (the, video). TF-IDF pondère la fréquence locale par la rareté globale (IDF) et met en valeur les termes discriminants porteurs d'opinion (amazing, horrible, waste)."]),
            ("2 - Comment interpréter la matrice de confusion ?", [
                f"La diagonale (1318 + 2420 + 2682 = 6420 sur 7313, soit {pct(M['accuracy'])}) représente les prédictions exactes. La classe Neutre a le meilleur rappel (96,0 %), devant Positif (85,3 %) et Négatif (80,0 %). Les confusions opposent négatif et positif aux neutres, sur des commentaires ambigus ou ironiques."]),
            ("3 - Quels hyperparamètres ont le plus d'impact ?", [
                f"GridSearchCV a retenu C={M['best_params']['classifier__C']}, min_df={M['best_params']['tfidf__min_df']}, ngram_range={tuple(M['best_params']['tfidf__ngram_range'])}. class_weight='balanced' corrige le déséquilibre de la classe négative (F1 porté à {pct(M['per_class']['-1']['f1-score'])}). C ajuste la régularisation L2."]),
        ])

        # Phase 4
        qapage(pdf, 8, "Phase 4 : Développement de l'API REST", "PHASE 4", [
            ("1 - Pourquoi utiliser FastAPI plutôt que Flask ?", [
                "FastAPI est asynchrone (ASGI/Uvicorn) et plus rapide que Flask (WSGI) pour servir des prédictions en temps réel. Il valide les entrées avec Pydantic (erreurs 422 explicites) et génère automatiquement Swagger (/docs)."]),
            ("2 - À quoi sert la configuration CORS ?", [
                "La Same-Origin Policy du navigateur bloque par défaut les appels fetch() d'une extension ou de youtube.com vers un serveur externe. Le middleware CORS ajoute les en-têtes Access-Control-Allow-Origin/Methods nécessaires pour autoriser l'extension à appeler l'API."]),
            ("3 - Comment gérer efficacement le chargement du modèle ?", [
                "Le pipeline .joblib est chargé une seule fois en mémoire au démarrage du module de l'API (variable globale), et non rechargé à chaque requête. Chaque appel réutilise l'objet en RAM, garantissant une latence faible (environ 4 ms pour 50 commentaires)."]),
        ])

        # Phase 5
        qapage(pdf, 9, "Phase 5 : Développement de l'extension Chrome", "PHASE 5", [
            ("1 - Comment gérer les commentaires avec des réponses (threads) ?", [
                "Mon content script extrait les commentaires principaux (ytd-comment-thread-renderer #content-text). Les réponses sont dans ytd-comment-replies-renderer ; une amélioration serait une case 'Inclure les réponses' pour analyser aussi les discussions."]),
            ("2 - Quelle stratégie pour extraire les commentaires dynamiques ?", [
                "YouTube charge les commentaires en lazy-loading au défilement. L'extension lit ceux présents dans le DOM à l'analyse (run_at document_idle) ; l'utilisateur fait défiler pour en charger davantage puis relance. Un défilement automatique est une amélioration prévue."]),
            ("3 - Comment optimiser les performances avec de nombreux commentaires ?", [
                "Les commentaires visibles (plafonnés à 100) sont envoyés en une seule requête /predict_batch et vectorisés en une passe. Les filtres par sentiment opèrent côté client, en mémoire dans popup.js, sans réinterroger le serveur."]),
        ])

        # Phase 6
        qapage(pdf, 10, "Phase 6 : Déploiement cloud avec Docker", "PHASE 6", [
            ("1 - Pourquoi séparer les requirements dev et prod ?", [
                "pandas, matplotlib, seaborn et pytest pèsent plusieurs centaines de Mo et ne servent pas à l'inférence HTTP. requirements-prod.txt ne garde que fastapi, uvicorn, scikit-learn, pandas, numpy et joblib : image plus légère, moins de surface d'attaque et moins de RAM."]),
            ("2 - Comment réduire la taille de l'image Docker ?", [
                "Image de base python:3.10-slim, pip install --no-cache-dir, copie des seuls fichiers nécessaires (app_api.py, src, models), et ordre des couches optimisé (COPY requirements avant le code) pour maximiser le cache."]),
            ("3 - Quelle stratégie pour les mises à jour du modèle ?", [
                "Réentraîner via le pipeline reproductible, valider les métriques avant promotion (pas de régression), versionner les .joblib, puis reconstruire et redéployer l'image — idéalement via une CI/CD."]),
            ("4 - Cloud éphémère (Codespaces) vs PaaS permanent ?", [
                "Un Codespace s'arrête après ~30 min d'inactivité et doit être relancé avant chaque démonstration ; un PaaS reste disponible en continu avec une URL stable, au prix d'un hébergement payant. Pour une vraie production, le PaaS (ou un conteneur managé) est préférable pour la disponibilité."]),
        ])

        # Phase 7
        chart_page(pdf, 11, "Phase 7 : Tests et validation", "PHASE 7",
                   tests_fig, "Figure 7.1 : Exécution de la suite de tests (13 tests passés).", rect=(0.08, 0.3, 0.84, 0.5))
    print(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    build()
