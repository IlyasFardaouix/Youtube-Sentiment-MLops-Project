"""Generate the reflection-questions report PDF (per TP phase) from real artifacts."""
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

NAVY, TEAL, ORANGE, INK, MUTED, PAPER = "#173b4a", "#23858b", "#d47b4d", "#24313a", "#687781", "#fbfaf7"
AUTHOR = "Ilyas Fardaoui"


def pct(x):
    return f"{x*100:.2f}".replace(".", ",") + " %"


def page(title, kicker=""):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(PAPER)
    if kicker:
        fig.text(0.08, 0.95, kicker, fontsize=12, color=TEAL, weight="bold")
    fig.text(0.08, 0.915, title, fontsize=16, weight="bold", color=NAVY)
    return fig


def foot(fig, n):
    fig.text(0.08, 0.035, f"TP3 - Projet ML Complet | {AUTHOR}", fontsize=8, color=MUTED)
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


def figpage(pdf, n, title, kicker, path, caption, make=None):
    fig = page(title, kicker)
    ax = fig.add_axes([0.14, 0.32, 0.72, 0.46])
    if make:
        make(ax)
    else:
        ax.axis("off")
        if path and path.exists():
            ax.imshow(mpimg.imread(path))
    fig.text(0.1, 0.26, caption, fontsize=9.5, color=MUTED)
    foot(fig, n)
    pdf.savefig(fig, facecolor=PAPER)
    plt.close(fig)


def confusion(ax):
    cm = M["confusion_matrix"]
    ax.imshow(cm, cmap="BuGn")
    labels = ["-1", "0", "1"]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    peak = max(max(r) for r in cm)
    for i, r in enumerate(cm):
        for j, v in enumerate(r):
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > peak / 2 else INK, fontsize=12)


def comparison(ax):
    c = M["candidates"]
    names = ["Logistic\nRegression", "Linear\nSVM"]
    vals = [c["logistic_regression"] * 100, c["linear_svm"] * 100]
    bars = ax.bar(names, vals, color=[TEAL, MUTED], width=0.5)
    ax.set_ylim(80, 92)
    ax.set_ylabel("Accuracy (%)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}%", ha="center", fontsize=11, weight="bold", color=NAVY)
    ax.set_title("Comparaison des modèles (accuracy brute)", fontsize=11, color=NAVY)


def build():
    with PdfPages(OUTPUT) as pdf:
        # cover
        cover = plt.figure(figsize=(8.27, 11.69))
        cover.patch.set_facecolor(PAPER)
        cover.text(0.08, 0.80, "TP3 - Projet Cloud Computing", fontsize=26, weight="bold", color=NAVY)
        cover.text(0.08, 0.745, "Analyse de sentiment des commentaires YouTube", fontsize=15, color=ORANGE)
        cover.text(0.08, 0.70, "Questions de réflexion", fontsize=14, color=TEAL, weight="bold")
        cover.text(0.08, 0.55, "Élément de Module : Industrie 4.0", fontsize=13, color=INK)
        cover.text(0.08, 0.50, f"Réalisé par : {AUTHOR}", fontsize=13, color=INK)
        cover.text(0.08, 0.455, "Encadré par : Prof. Maria Zemzami", fontsize=13, color=INK)
        cover.text(0.08, 0.37, "Université Mohamed V - ENSAM Rabat\nFilière INDIA | Année universitaire 2026/27", fontsize=12, color=MUTED, linespacing=1.6)
        foot(cover, 1)
        pdf.savefig(cover, facecolor=PAPER)
        plt.close(cover)

        # Phase 1
        qapage(pdf, 2, "Phase 1 : Configuration de l'environnement MLOps", "PHASE 1", [
            ("1 - Pourquoi séparer les données raw et processed ?", [
                "Le dossier data/raw conserve les données brutes intactes : si le script de nettoyage change (gestion de la ponctuation, des symboles), on régénère data/processed sans retélécharger ni altérer la source.",
                "Cela garantit un flux unidirectionnel (raw -> nettoyage -> processed) et permet à l'entraînement de charger directement les splits nettoyés sans rejouer le nettoyage à chaque expérimentation."]),
            ("2 - Quels fichiers ne doivent jamais être commités sur Git ?", [
                "L'environnement virtuel (.venv/), les secrets (.env, tokens), les datasets volumineux (data/raw et data/processed/*.csv), les caches et artefacts (__pycache__/, *.pyc, logs/), ainsi que les gros modèles. Mon .gitignore couvre précisément ces cas."]),
            ("3 - Quelle est l'importance d'un environnement virtuel ?", [
                "Il isole les dépendances du projet (versions de scikit-learn, FastAPI, etc.) pour éviter les conflits entre projets, et protège l'installation Python globale du système. Il rend aussi l'environnement reproductible via requirements.txt."]),
        ])

        # Phase 2
        total = 36562
        intro2 = ("Après nettoyage : 36 562 commentaires (train 29 249 / test 7 313). Répartition : positif 15 723 (43,0 %), "
                  "neutre 12 606 (34,5 %), négatif 8 233 (22,5 %) — plus de 1 000 par classe comme exigé.")
        qapage(pdf, 3, "Phase 2 : Pipeline de données et préparation", "PHASE 2", [
            ("1 - Pourquoi nettoyer les URLs et mentions ?", [
                "Les URLs et mentions sont quasi uniques et créent des milliers de tokens rares, sans valeur sémantique, qui gonflent la matrice TF-IDF. Les remplacer par des tokens (URL, USER, HASHTAG) réduit le bruit et la dimensionnalité et limite le surapprentissage, ce qui aide le modèle à généraliser aux commentaires YouTube."]),
            ("2 - Comment gérer les emojis dans le texte ?", [
                "Dans mon nettoyage actuel, la normalisation (suppression des caractères non alphanumériques hors ! ? ') retire les emojis avec les autres symboles. C'est un choix de simplicité qui privilégie le signal textuel. Une amélioration identifiée serait de les convertir en tokens sémantiques (emoji_positif, emoji_négatif) avant la normalisation pour conserver ce signal émotionnel, utile sur les commentaires courts."]),
            ("3 - Quel ratio train/test est approprié ?", [
                "Un split stratifié 80 % / 20 % (random_state=42) : environ 29 249 commentaires pour apprendre le vocabulaire TF-IDF et la validation croisée, et 7 313 pour une évaluation finale fiable (dont plus de 1 600 pour la classe minoritaire négative). La stratification préserve la distribution des classes."]),
        ], intro=intro2)

        # Phase 3 - figures
        figpage(pdf, 4, "Phase 3 : Développement et entraînement du modèle", "PHASE 3", None,
                "Figure 3.1 : Matrice de confusion (jeu de test, 7 313 commentaires).", make=confusion)
        figpage(pdf, 5, "Phase 3 : Comparaison des modèles", "PHASE 3", None,
                "Figure 3.2 : Logistic Regression retenue pour ses probabilités (predict_proba).", make=comparison)
        cm = M["confusion_matrix"]
        qapage(pdf, 6, "Phase 3 : Questions de réflexion", "PHASE 3", [
            ("1 - Pourquoi utiliser TF-IDF plutôt que Count Vectorizer ?", [
                "CountVectorizer donne un poids élevé aux mots fréquents mais peu informatifs (the, video, think). TF-IDF pondère la fréquence locale par la rareté globale (IDF) et met en valeur les termes discriminants porteurs d'opinion (amazing, horrible, waste)."]),
            ("2 - Comment interpréter la matrice de confusion ?", [
                f"La diagonale (1318 + 2420 + 2682 = 6420 sur 7313, soit {pct(M['accuracy'])}) représente les prédictions exactes. La classe Neutre a le meilleur rappel (96,0 %), devant Positif (85,3 %) et Négatif (80,0 %). Les confusions principales opposent négatif et positif aux neutres, souvent sur des commentaires ambigus ou ironiques."]),
            ("3 - Quels hyperparamètres ont le plus d'impact ?", [
                f"GridSearchCV a retenu C={M['best_params']['classifier__C']}, min_df={M['best_params']['tfidf__min_df']}, ngram_range={tuple(M['best_params']['tfidf__ngram_range'])}. class_weight='balanced' corrige le déséquilibre de la classe négative (F1 porté à {pct(M['per_class']['-1']['f1-score'])}). Le paramètre C ajuste la régularisation L2 et ngram_range contrôle la prise en compte des expressions composées."]),
        ])

        # Phase 4
        qapage(pdf, 7, "Phase 4 : Développement de l'API REST", "PHASE 4", [
            ("1 - Pourquoi utiliser FastAPI plutôt que Flask ?", [
                "FastAPI est asynchrone (ASGI/Uvicorn) et plus rapide que Flask (WSGI) pour servir des prédictions en temps réel. Il valide nativement les entrées avec Pydantic (erreurs 422 explicites) et génère automatiquement la documentation OpenAPI (Swagger /docs)."]),
            ("2 - À quoi sert la configuration CORS ?", [
                "La Same-Origin Policy du navigateur bloque par défaut les appels fetch() d'une extension Chrome ou de youtube.com vers un serveur externe. Le middleware CORS ajoute les en-têtes Access-Control-Allow-Origin/Methods nécessaires pour autoriser l'extension à appeler l'API."]),
            ("3 - Comment gérer efficacement le chargement du modèle ?", [
                "Le pipeline .joblib est chargé une seule fois en mémoire au démarrage du module de l'API (variable globale), et non rechargé depuis le disque à chaque requête. Chaque appel /predict_batch réutilise l'objet déjà en RAM, ce qui garantit une latence faible (environ 4 ms pour 50 commentaires)."]),
        ])

        # Phase 5
        qapage(pdf, 8, "Phase 5 : Développement de l'extension Chrome", "PHASE 5", [
            ("1 - Comment gérer les commentaires avec des réponses (threads) ?", [
                "Mon content script extrait les commentaires principaux (ytd-comment-thread-renderer #content-text). Les réponses sont imbriquées dans ytd-comment-replies-renderer ; une amélioration serait d'ajouter une case 'Inclure les réponses' pour choisir d'analyser aussi les discussions."]),
            ("2 - Quelle stratégie pour extraire les commentaires dynamiques ?", [
                "YouTube charge les commentaires en lazy-loading au défilement. L'extension lit les commentaires présents dans le DOM au moment de l'analyse (run_at document_idle) ; l'utilisateur fait défiler pour en charger davantage puis relance l'analyse. Un défilement automatique vers #comments est une amélioration prévue."]),
            ("3 - Comment optimiser les performances avec de nombreux commentaires ?", [
                "Les commentaires visibles (plafonnés à 100 par extraction) sont envoyés en une seule requête /predict_batch et vectorisés en une passe. Les filtres par sentiment opèrent côté client, en mémoire dans popup.js, sans réinterroger le serveur."]),
        ])

        # Phase 6
        qapage(pdf, 9, "Phase 6 : Déploiement cloud avec Docker", "PHASE 6", [
            ("1 - Pourquoi séparer les requirements dev et prod ?", [
                "requirements.txt contient les outils de dev (pytest, matplotlib, seaborn) ; requirements-prod.txt ne garde que l'exécution (fastapi, uvicorn, scikit-learn, pandas, numpy, joblib). L'image Docker n'installe que le prod, ce qui la rend plus légère et plus sûre."]),
            ("2 - Comment réduire la taille de l'image Docker ?", [
                "Image de base légère (python:3.10-slim), installation des seuls requirements de production, copie uniquement des fichiers nécessaires (app, src, models), et --no-cache-dir pour pip. L'ordre des couches est optimisé pour profiter du cache."]),
            ("3 - Quelle stratégie pour les mises à jour du modèle ?", [
                "Réentraîner via le pipeline reproductible, versionner les nouveaux .joblib et métriques, puis reconstruire et redéployer l'image. Un suivi des métriques permet de valider qu'une nouvelle version ne régresse pas avant mise en production."]),
            ("4 - Cloud éphémère (Codespaces) vs hébergement PaaS permanent ?", [
                "Un Codespace est éphémère : il s'arrête après environ 30 min d'inactivité et son URL n'est active que lorsqu'il tourne, ce qui impose de le relancer avant chaque démonstration. Un PaaS permanent reste disponible en continu avec une URL stable, au prix d'un hébergement payant. Pour une vraie mise en production, le PaaS (ou un conteneur managé) est préférable pour la disponibilité."]),
        ])
    print(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    build()
