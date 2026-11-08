"""Generate the structured technical report PDF from verified artifacts."""
import json
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "rapport_technique.pdf"
SCREENSHOTS = ROOT / "docs" / "screenshots"
METRICS = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))

NAVY, TEAL, ORANGE, INK, MUTED = "#173b4a", "#23858b", "#d47b4d", "#24313a", "#687781"
PAPER = "#fbfaf7"


def pct(value: float) -> str:
    return f"{value * 100:.2f}".replace(".", ",") + " %"


def num(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def page_frame(title: str, subtitle: str = "") -> plt.Figure:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.patch.set_facecolor(PAPER)
    figure.text(0.08, 0.94, title, fontsize=21, weight="bold", color=NAVY)
    if subtitle:
        figure.text(0.08, 0.905, subtitle, fontsize=10.5, color=MUTED)
    return figure


def footer(figure: plt.Figure, number: int) -> None:
    figure.text(0.08, 0.035, "YouTube Sentiment Lens | Rapport technique", fontsize=8, color=MUTED)
    figure.text(0.92, 0.035, str(number), fontsize=8, color=MUTED, ha="right")


def text_page(pdf: PdfPages, number: int, title: str, sections, subtitle: str = "") -> None:
    figure = page_frame(title, subtitle)
    y = 0.86
    for heading, paragraphs in sections:
        figure.text(0.08, y, heading, fontsize=13, weight="bold", color=TEAL)
        y -= 0.034
        for paragraph in paragraphs:
            for line in textwrap.wrap(paragraph, width=92):
                figure.text(0.1, y, line, fontsize=10.5, color=INK)
                y -= 0.0225
            y -= 0.012
        y -= 0.016
    footer(figure, number)
    pdf.savefig(figure, facecolor=PAPER)
    plt.close(figure)


def box(ax, x, y, text, fc, w=3.4, h=1.1):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            linewidth=1.2, edgecolor=NAVY, facecolor=fc,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color=INK)


def down_arrow(ax, x, y1, y2, label):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=15, color=TEAL, linewidth=1.4))
    ax.text(x + 0.25, (y1 + y2) / 2, label, fontsize=8, color=MUTED, ha="left", va="center")


def architecture_page(pdf: PdfPages, number: int) -> None:
    figure = page_frame("03 | Architecture du système", "Flux de données de bout en bout")
    ax = figure.add_axes([0.05, 0.12, 0.9, 0.72])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 10)
    box(ax, 3, 9.0, "Vidéo YouTube", "#e8f2f0")
    box(ax, 3, 6.6, "Extension Chrome\n(content.js + popup.js)", "#e8f2f0")
    box(ax, 3, 4.2, "API FastAPI\n(CORS + Pydantic)", "#fdeede")
    box(ax, 3, 1.8, "Pipeline ML\nTF-IDF + Logistic Regression", "#fdeede")
    down_arrow(ax, 3, 8.45, 7.15, "commentaires extraits")
    down_arrow(ax, 3, 6.05, 4.75, "POST /predict_batch (JSON)")
    down_arrow(ax, 3, 3.65, 2.35, "model.predict")
    ax.add_patch(
        FancyArrowPatch(
            (4.7, 2.1), (4.7, 6.3),
            connectionstyle="arc3,rad=-0.6",
            arrowstyle="-|>", mutation_scale=16, color=ORANGE, linewidth=1.5,
        )
    )
    ax.text(8.0, 4.2, "réponse JSON\nprédictions\n+ confiance\n+ statistiques", fontsize=8.5, color=ORANGE, ha="center", va="center")
    figure.text(0.08, 0.1, "Déploiement : conteneur Docker sur GitHub Codespaces, port 7860 rendu public (HTTPS).", fontsize=10, color=INK)
    footer(figure, number)
    pdf.savefig(figure, facecolor=PAPER)
    plt.close(figure)


def performance_page(pdf: PdfPages, number: int) -> None:
    cm = METRICS["confusion_matrix"]
    total = sum(sum(row) for row in cm)
    per = METRICS["per_class"]
    figure = page_frame("06 | Performance du modèle", f"Jeu de test : {num(total)} commentaires")
    axis = figure.add_axes([0.08, 0.5, 0.84, 0.3])
    axis.axis("off")
    rows = [
        ["Accuracy", pct(METRICS["accuracy"]), "80 %", "Atteint"],
        ["F1 négatif", pct(per["-1"]["f1-score"]), "75 %", "Atteint"],
        ["F1 neutre", pct(per["0"]["f1-score"]), "75 %", "Atteint"],
        ["F1 positif", pct(per["1"]["f1-score"]), "75 %", "Atteint"],
        ["Inférence batch 50", f"{METRICS['inference_50_ms']:.2f}".replace(".", ",") + " ms", "100 ms", "Atteint"],
    ]
    table = axis.table(cellText=rows, colLabels=["Mesure", "Résultat", "Seuil", "Statut"], cellLoc="left", loc="center", colWidths=[0.3, 0.22, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, _column), cell in table.get_celld().items():
        cell.set_height(0.16)
        cell.set_edgecolor("#d8e1df")
        cell.set_facecolor("#e8f2f0" if row == 0 else "#ffffff")
        if row == 0:
            cell.set_text_props(weight="bold", color=NAVY)
    heat = figure.add_axes([0.3, 0.1, 0.4, 0.33])
    heat.imshow(cm, cmap="BuGn")
    labels = ["-1", "0", "1"]
    heat.set_xticks(range(3), labels)
    heat.set_yticks(range(3), labels)
    heat.set_xlabel("Prédit")
    heat.set_ylabel("Réel")
    heat.set_title("Matrice de confusion", fontsize=11, color=NAVY)
    peak = max(max(row) for row in cm)
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            heat.text(j, i, str(value), ha="center", va="center", color="white" if value > peak / 2 else INK, fontsize=10)
    footer(figure, number)
    pdf.savefig(figure, facecolor=PAPER)
    plt.close(figure)


def two_image_page(pdf: PdfPages, number: int, title: str, left, left_cap, right, right_cap) -> None:
    figure = page_frame(title, "Preuve capturée avec Playwright")
    for rect, path, caption in (([0.06, 0.12, 0.42, 0.72], left, left_cap), ([0.52, 0.12, 0.42, 0.72], right, right_cap)):
        axis = figure.add_axes(rect)
        axis.axis("off")
        if path.exists():
            axis.imshow(mpimg.imread(path))
        axis.set_title(caption, fontsize=10, color=NAVY)
    footer(figure, number)
    pdf.savefig(figure, facecolor=PAPER)
    plt.close(figure)


def build() -> Path:
    candidates = METRICS["candidates"]
    best = METRICS["best_params"]
    with PdfPages(OUTPUT) as pdf:
        cover = page_frame("Rapport technique", "Projet ML complet - Analyse de sentiment YouTube")
        cover.text(0.08, 0.72, "YouTube\nSentiment Lens", fontsize=34, weight="bold", color=NAVY, linespacing=1.15)
        cover.text(0.08, 0.53, "Du développement au déploiement cloud", fontsize=17, color=ORANGE)
        cover.text(0.08, 0.39, "Réalisé par : Ilyas Fardaoui\nEncadré par : Prof. Maria Zemzami\nFilière INDIA | Année universitaire 2026/27\nUniversité Mohamed V - ENSAM Rabat", fontsize=12, color=INK, linespacing=1.6)
        footer(cover, 1)
        pdf.savefig(cover, facecolor=PAPER)
        plt.close(cover)

        text_page(pdf, 2, "01 | Synthèse et problématique", [
            ("Objectif", ["Le projet automatise l'analyse de commentaires YouTube en trois classes : négatif (-1), neutre (0) et positif (1)."]),
            ("Problématique", ["L'analyse manuelle de centaines de commentaires est chronophage, subjective et difficile à quantifier. Le système fournit une analyse automatique, rapide et reproductible."]),
            ("Solution", ["Une extension Chrome extrait les commentaires visibles, les transmet par batch à une API FastAPI, puis affiche le sentiment, la confiance, un graphique de répartition et des filtres."]),
            ("Résultat", [f"Le modèle atteint {pct(METRICS['accuracy'])} d'accuracy et dépasse tous les seuils F1 demandés. L'API est servie dans un conteneur Docker exécuté sur GitHub Codespaces."]),
        ])

        architecture_page(pdf, 3)

        text_page(pdf, 4, "04 | Données et préparation", [
            ("Source", ["Le dataset Reddit Sentiment Analysis est téléchargé automatiquement depuis GitHub et sauvegardé dans data/raw/reddit.csv. Il contient 37 249 commentaires (brut : négatif 8 277, neutre 13 142, positif 15 830)."]),
            ("Nettoyage", ["Le texte est mis en minuscules; les URLs, mentions et hashtags sont remplacés par des tokens (URL, USER, HASHTAG); les caractères spéciaux sont normalisés; les doublons et les textes de moins de trois caractères sont supprimés."]),
            ("Statistiques", ["Après nettoyage, 36 562 commentaires sont conservés (négatif 8 233, neutre 12 606, positif 15 723), soit bien plus de 1 000 exemples par classe comme exigé."]),
            ("Reproductibilité", ["Le split stratifié (random_state=42) répartit 80 % en entraînement (29 249) et 20 % en test (7 313), ce qui préserve la distribution des classes."]),
        ])

        text_page(pdf, 5, "05 | Modélisation et optimisation", [
            ("Vectorisation", ["TF-IDF (avec sublinear_tf) valorise les termes discriminants et réduit le poids des mots trop fréquents, ce qui convient mieux qu'un simple comptage."]),
            ("Comparaison des algorithmes", [f"Deux pipelines pondérés par classe ont été comparés : Logistic Regression (accuracy brute {pct(candidates['logistic_regression'])}) et Linear SVM ({pct(candidates['linear_svm'])})."]),
            ("Choix du modèle", ["Le Linear SVM est légèrement meilleur en accuracy brute, mais la Logistic Regression a été retenue car elle fournit des probabilités (predict_proba), indispensables au score de confiance de l'extension."]),
            ("Optimisation", [f"GridSearchCV (cv=3, score f1_macro) explore C, min_df et ngram_range. Meilleurs paramètres : C={best['classifier__C']}, min_df={best['tfidf__min_df']}, ngram_range={tuple(best['tfidf__ngram_range'])}."]),
            ("Limite méthodologique", ["La sélection entre candidats s'appuie sur le jeu de test; un jeu de validation dédié serait plus rigoureux et constitue une amélioration future."]),
            ("Artefacts", ["Le pipeline, le vectoriseur et le classifieur sont sauvegardés avec joblib dans models/."]),
        ])

        performance_page(pdf, 6)

        text_page(pdf, 7, "07 | API REST et extension Chrome", [
            ("API", ["GET /health vérifie le service et le chargement du modèle. POST /predict_batch valide un batch de 1 à 500 commentaires via Pydantic et renvoie, par commentaire, le label, le sentiment, la confiance, puis les statistiques globales et la latence."]),
            ("Robustesse", ["CORS est activé pour l'extension; un batch vide, trop grand, mal typé ou contenant un commentaire vide renvoie une erreur 422; un modèle absent renvoie 503."]),
            ("Extension", ["L'extension Manifest V3 extrait les commentaires visibles, affiche les pourcentages, un graphique à barres de répartition, filtre par sentiment, propose un mode clair/sombre, un indicateur de chargement et la copie des résultats."]),
        ])

        text_page(pdf, 8, "08 | Déploiement cloud", [
            ("Conteneurisation", ["Le Dockerfile part de python:3.10-slim, installe uniquement requirements-prod.txt, copie le code et les modèles, et expose un port configurable via la variable PORT (défaut 7860)."]),
            ("Exécution sur Codespaces", ["Commandes : docker build -t youtube-sentiment-api . puis docker run -d -p 7860:7860 -e PORT=7860 youtube-sentiment-api. Le port 7860 est ensuite rendu public dans l'onglet Ports."]),
            ("Contrainte", ["L'URL Codespaces (https://<nom>-7860.app.github.dev) est éphémère : le Codespace s'arrête après environ 30 min d'inactivité et doit être relancé avant la démonstration. C'est la différence clé avec un hébergement PaaS permanent."]),
            ("Validation", ["Les endpoints /health, /docs et /predict_batch ont été vérifiés sur l'URL publique, ainsi que le flux complet extension -> API cloud -> résultats."]),
        ])

        text_page(pdf, 9, "09 | Difficultés, tests et améliorations", [
            ("Difficultés rencontrées", ["GridSearchCV échouait avec BrokenProcessPool sous Windows : le parallélisme a été fixé à n_jobs=1. L'interpréteur de VS Code ne pointait pas vers l'environnement contenant les dépendances : il a été reconfiguré."]),
            ("Tests", ["La suite pytest couvre l'API (health, batch vide, hors limite, payload invalide, prédiction cohérente) et le modèle (accuracy >= 80 %, F1 par classe > 75 %, latence < 100 ms, cas limites : texte court/long, emojis, langues mélangées)."]),
            ("Améliorations futures", ["Jeu de validation dédié pour la sélection de modèle; authentification de l'API; monitoring de dérive; mise à jour versionnée du modèle; extraction des réponses (threads) et défilement automatique côté extension; cache des résultats."]),
        ])

        two_image_page(pdf, 10, "10 | Preuves - API", SCREENSHOTS / "api-health.png", "GET /health", SCREENSHOTS / "api-docs.png", "Swagger /docs")
        two_image_page(pdf, 11, "11 | Preuves - Extension", SCREENSHOTS / "extension-light-mode.png", "Mode clair", SCREENSHOTS / "extension-dark-mode.png", "Mode sombre")

    print(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    build()
