"""Generate a structured technical report PDF with verified evidence."""
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "rapport_technique.pdf"
SCREENSHOTS = ROOT / "docs" / "screenshots"
NAVY, TEAL, ORANGE, INK, MUTED = "#173b4a", "#23858b", "#d47b4d", "#24313a", "#687781"


def page_frame(title: str, subtitle: str = "") -> plt.Figure:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.patch.set_facecolor("#fbfaf7")
    figure.text(0.08, 0.94, title, fontsize=22, weight="bold", color=NAVY)
    if subtitle:
        figure.text(0.08, 0.905, subtitle, fontsize=10.5, color=MUTED)
    return figure


def footer(figure: plt.Figure, number: int) -> None:
    figure.text(0.08, 0.035, "YouTube Sentiment Lens | Rapport technique", fontsize=8, color=MUTED)
    figure.text(0.92, 0.035, str(number), fontsize=8, color=MUTED, ha="right")


def text_page(pdf: PdfPages, number: int, title: str, sections: list[tuple[str, list[str]]]) -> None:
    figure = page_frame(title)
    y = 0.84
    for heading, paragraphs in sections:
        figure.text(0.08, y, heading, fontsize=14, weight="bold", color=TEAL)
        y -= 0.045
        for paragraph in paragraphs:
            figure.text(0.1, y, paragraph, fontsize=10.5, color=INK, va="top", wrap=True)
            y -= 0.06 + 0.012 * (len(paragraph) // 105)
        y -= 0.02
    footer(figure, number)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def image_page(pdf: PdfPages, number: int, path: Path, title: str) -> None:
    figure = page_frame(title, "Preuve capturée avec Playwright")
    axis = figure.add_axes([0.08, 0.11, 0.84, 0.72])
    axis.axis("off")
    if path.exists():
        axis.imshow(mpimg.imread(path))
    footer(figure, number)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


with PdfPages(OUTPUT) as pdf:
    cover = page_frame("Rapport technique", "Projet ML complet - Analyse de sentiment YouTube")
    cover.text(0.08, 0.72, "YouTube\nSentiment Lens", fontsize=34, weight="bold", color=NAVY, linespacing=1.15)
    cover.text(0.08, 0.53, "Du développement au déploiement cloud", fontsize=17, color=ORANGE)
    cover.text(0.08, 0.39, "Filière INDIA | Année universitaire 2026/27\nUniversité Mohamed V - ENSAM Rabat\nProf. Maria Zemzami", fontsize=12, color=INK, linespacing=1.6)
    footer(cover, 1)
    pdf.savefig(cover, bbox_inches="tight")
    plt.close(cover)

    text_page(pdf, 2, "01 | Synthèse du projet", [("Objectif", ["Le projet automatise l'analyse de commentaires YouTube en trois classes : négatif (-1), neutre (0) et positif (1)."]), ("Solution", ["Une extension Chrome extrait les commentaires visibles, les transmet par batch à une API FastAPI et affiche le sentiment, la confiance et les statistiques."]), ("Résultat", ["Le modèle atteint 87,79 % d'accuracy et dépasse les seuils F1 demandés. L'API est exécutée dans Docker sur GitHub Codespaces."])])
    text_page(pdf, 3, "02 | Problème et architecture", [("Problématique", ["L'analyse manuelle de centaines de commentaires est lente, subjective et difficile à quantifier. Le système propose une analyse automatique et reproductible."]), ("Architecture", ["Frontend : extension Chrome Manifest V3. Backend : FastAPI avec Pydantic et CORS. ML Engine : TF-IDF + Logistic Regression. Cloud : Docker dans Codespaces, port public 7860."]), ("Flux", ["YouTube -> extraction -> POST /predict_batch -> modèle ML -> prédictions, confiance et statistiques."])])
    text_page(pdf, 4, "03 | Données et préparation", [("Source", ["Le dataset Reddit Sentiment Analysis est téléchargé automatiquement depuis GitHub et sauvegardé dans data/raw/reddit.csv."]), ("Nettoyage", ["URLs, mentions et hashtags sont remplacés par des tokens; les caractères spéciaux sont normalisés; les textes vides et doublons sont supprimés."]), ("Statistiques", ["37 149 commentaires téléchargés; 36 562 conservés après nettoyage; 8 277 négatifs, 13 042 neutres et 15 830 positifs."]), ("Reproductibilité", ["Split stratifié 80/20 avec random_state=42."])])
    text_page(pdf, 5, "04 | Modélisation et optimisation", [("Vectorisation", ["TF-IDF valorise les termes discriminants et réduit l'influence des mots trop fréquents."]), ("Algorithmes", ["Logistic Regression pondérée par classe comparée à Linear SVM. La Logistic Regression est retenue pour ses probabilités de confiance."]), ("Optimisation", ["GridSearchCV teste C, min_df et ngram_range. Meilleurs paramètres : C=2.0, min_df=2, ngram_range=(1,1)."]), ("Artefacts", ["Pipeline, vectoriseur et classifieur sauvegardés avec joblib dans models/."])])

    performance = page_frame("05 | Performance du modèle", "Test set : 7 313 commentaires")
    axis = performance.add_axes([0.08, 0.34, 0.84, 0.42])
    axis.axis("off")
    table = axis.table(cellText=[["Accuracy", "87,79 %", "80 %", "Atteint"], ["F1 négatif", "81,18 %", "75 %", "Atteint"], ["F1 neutre", "90,79 %", "75 %", "Atteint"], ["F1 positif", "88,69 %", "75 %", "Atteint"], ["Inférence batch 50", "3,57 ms", "100 ms", "Atteint"]], colLabels=["Mesure", "Résultat", "Seuil", "Statut"], cellLoc="left", loc="center", colWidths=[0.28, 0.22, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_height(0.13)
        cell.set_edgecolor("#d8e1df")
        cell.set_facecolor("#e8f2f0" if row == 0 else "#ffffff")
        if row == 0:
            cell.set_text_props(weight="bold", color=NAVY)
    performance.text(0.1, 0.22, "Matrice de confusion : [1318,149,180] / [60,2420,41] / [222,241,2682]. Les confusions principales concernent les commentaires ambigus.", fontsize=10.5, color=INK, wrap=True)
    footer(performance, 6)
    pdf.savefig(performance, bbox_inches="tight")
    plt.close(performance)

    text_page(pdf, 7, "06 | API REST et extension Chrome", [("API", ["GET /health vérifie le service. POST /predict_batch accepte 1 à 500 commentaires et renvoie label, sentiment, confiance, comptes et latence."]), ("Extension", ["Extraction des commentaires visibles, statistiques en pourcentage, filtres, mode clair/sombre et copie des résultats."]), ("Validation", ["Swagger /docs et OpenAPI vérifiés sur l'URL publique. Le flux réel YouTube -> extension -> API a été testé avec 19 commentaires."])])
    text_page(pdf, 8, "07 | Déploiement, tests et livrables", [("Docker et Codespaces", ["Image python:3.10-slim, requirements-prod.txt, code et modèles. Commandes : docker build -t youtube-sentiment-api . puis docker run -d --name youtube-sentiment-api -p 7860:7860 -e PORT=7860 youtube-sentiment-api."]), ("URL publique", ["https://curly-happiness-7vpr5wr6jgq9fpwjr-7860.app.github.dev. Codespaces est éphémère et doit rester actif pendant la démonstration."]), ("Tests", ["3 tests automatisés passent. Docker build, health, OpenAPI, prédiction publique, syntaxe JavaScript et contrôles Playwright sont validés."]), ("Livrables et conclusion", ["Dépôt GitHub, modèles joblib, rapport PDF, captures et chrome-extension.zip. Les critères ML et les fonctionnalités demandées sont respectés."])])

    image_page(pdf, 9, SCREENSHOTS / "api-health.png", "08 | Preuve API : health")
    image_page(pdf, 10, SCREENSHOTS / "api-docs.png", "09 | Preuve API : Swagger")
    image_page(pdf, 11, SCREENSHOTS / "extension-light-mode.png", "10 | Preuve extension : mode clair")
    image_page(pdf, 12, SCREENSHOTS / "extension-dark-mode.png", "11 | Preuve extension : mode sombre")

print(OUTPUT)
