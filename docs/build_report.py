"""Build the technical report PDF from verified project artifacts."""
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "rapport_technique.pdf"
SCREENSHOTS = ROOT / "docs" / "screenshots"

PAGES = [
    ("Rapport technique", ["YouTube Sentiment Lens", "Analyse de sentiment YouTube avec TF-IDF, Logistic Regression, FastAPI, Chrome et Docker", "Projet INDIA - Année 2026/27"]),
    ("1. Problématique et architecture", ["Objectif : analyser automatiquement les commentaires YouTube en temps réel.", "Flux : Extension Chrome -> API FastAPI -> pipeline ML -> résultats et statistiques.", "Déploiement : Docker exécuté dans GitHub Codespaces, port public 7860.", "Versionnement : dépôt GitHub sur la branche main."]),
    ("2. Données et préparation", ["Source : Reddit Sentiment Analysis.", "37 149 commentaires téléchargés; 36 562 conservés après nettoyage.", "Labels : -1 négatif, 0 neutre, 1 positif.", "Split stratifié reproductible : 80 % train / 20 % test.", "Nettoyage : URLs, mentions, hashtags et caractères spéciaux normalisés."]),
    ("3. Modèle et optimisation", ["Vectorisation TF-IDF avec Logistic Regression pondérée par classe.", "Comparaison effectuée avec Linear SVM.", "GridSearchCV utilisé sur C, min_df et ngram_range.", "Meilleurs paramètres : C=2.0, min_df=2, ngram_range=(1,1)."]),
    ("4. Résultats", ["Accuracy : 87,79 % (seuil demandé : 80 %).", "F1 négatif : 81,18 %.", "F1 neutre : 90,79 %.", "F1 positif : 88,69 %.", "F1 macro : 86,89 %.", "Inférence de 50 commentaires : 3,57 ms (seuil : 100 ms)."]),
    ("5. API et extension", ["GET /health vérifie le chargement du modèle.", "POST /predict_batch valide un batch de 1 à 500 commentaires.", "La réponse contient le label, le sentiment, la confiance et les statistiques.", "L'extension extrait les commentaires visibles, propose des filtres, le mode sombre et la copie."]),
    ("6. Tests et déploiement", ["Tests automatisés : 3 passed.", "Docker build validé localement.", "Codespaces : port 7860 rendu public.", "Health public vérifié : status=ok, model_loaded=true.", "Swagger /docs vérifié avec les deux endpoints."]),
    ("7. Captures et conclusion", ["Les captures Playwright jointes montrent /health, Swagger et les modes clair/sombre de l'extension.", "Le projet respecte les seuils ML et les exigences fonctionnelles.", "Limite : Codespaces est éphémère; il doit rester actif pendant la démonstration.", "Améliorations : monitoring, authentification, cache et mise à jour versionnée du modèle."]),
]


def text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.92, title, fontsize=22, weight="bold", color="#173b3f")
    y = 0.84
    for line in lines:
        figure.text(0.1, y, line, fontsize=13, va="top", wrap=True)
        y -= 0.075
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def image_page(pdf: PdfPages, path: Path, title: str) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.94, title, fontsize=20, weight="bold", color="#173b3f")
    figure.figimage(mpimg.imread(path), xo=80, yo=180, resize=True)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


with PdfPages(OUTPUT) as pdf:
    for title, lines in PAGES:
        text_page(pdf, title, lines)
    for path, title in [(SCREENSHOTS / "api-health.png", "Capture Playwright - API health"), (SCREENSHOTS / "api-docs.png", "Capture Playwright - Swagger"), (SCREENSHOTS / "extension-light-mode.png", "Capture Playwright - Extension"), (SCREENSHOTS / "extension-dark-mode.png", "Capture Playwright - Mode sombre")]:
        if path.exists():
            image_page(pdf, path, title)

print(OUTPUT)
