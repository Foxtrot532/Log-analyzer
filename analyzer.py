#!/usr/bin/env python3
"""
analyzer.py
-----------
AI-Powered Log Analyzer & Support Ticket Classifier

Features:
- Regex-based log analyzer for quick error frequency insights
- ML-based ticket classifier (TF-IDF + MultinomialNB)
- Colorized CLI output for professional presentation
- Saves trained model (model.pkl) for reuse
"""

import os
import re
import joblib
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from colorama import init, Fore, Style

init(autoreset=True)

MODEL_FILE = "model.pkl"
DATA_FILE = "tickets.csv"

# -----------------------
#  TRAIN MODEL
# -----------------------
def train_model():
    if not os.path.exists(DATA_FILE):
        print(Fore.RED + f"[!] Dataset file '{DATA_FILE}' not found.")
        return

    df = pd.read_csv(DATA_FILE)
    X = df["description"]
    y = df["category"]

    # Stratified split for balanced category distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)

    print(Fore.GREEN + "[+] Model trained and saved as 'model.pkl'\n")

    y_pred = model.predict(X_test)
    print(Fore.CYAN + "--- Training Report ---")
    print(Fore.WHITE + classification_report(y_test, y_pred))
# -----------------------
#  PREDICT CATEGORY
# -----------------------
def predict_ticket(ticket_text):
    if not os.path.exists(MODEL_FILE):
        print(Fore.RED + "[!] Model not found. Train it first using --train.")
        return None

    model = joblib.load(MODEL_FILE)
    pred = model.predict([ticket_text])[0]

    # Top 3 probable categories (for fun)
    proba = model.predict_proba([ticket_text])[0]
    classes = model.classes_
    top3 = sorted(list(zip(classes, proba)), key=lambda x: x[1], reverse=True)[:3]

    print(Fore.CYAN + "\n--- Ticket Classification ---")
    print(Fore.GREEN + f"[+] Predicted Category: {pred}\n")
    print(Fore.YELLOW + "Top 3 Probabilities:")
    for cat, p in top3:
        print(f"   - {cat}: {p*100:.2f}%")

    return pred


# -----------------------
#  LOG ANALYZER
# -----------------------
def analyze_log(file_path):
    if not os.path.exists(file_path):
        print(Fore.RED + f"[!] Log file '{file_path}' not found.")
        return

    with open(file_path, "r", errors="ignore") as f:
        content = f.read()

    keywords = ["ERROR", "FAIL", "CRASH", "TIMEOUT", "DENIED", "DISCONNECT"]
    count = {k: len(re.findall(k, content, re.IGNORECASE)) for k in keywords}
    probable_category = "Unknown"

    # Quick heuristic (optional)
    if count["ERROR"] > 0 or count["CRASH"] > 0:
        probable_category = "Software Bug"
    elif count["TIMEOUT"] > 0 or count["DISCONNECT"] > 0:
        probable_category = "Network Issue"
    elif count["DENIED"] > 0:
        probable_category = "Authentication Issue"
    elif count["FAIL"] > 0:
        probable_category = "Storage Issue"

    print(Fore.CYAN + "\n--- Log Analysis Summary ---")
    for k, v in count.items():
        if v > 0:
            color = Fore.RED if k in ["ERROR", "CRASH", "FAIL"] else Fore.YELLOW
            bar = "█" * v
            print(f"{color}{k:<12}: {v}  {bar}")

    print(Fore.GREEN + f"\n[+] Probable Log Category: {probable_category}\n")

    return probable_category


# -----------------------
#  MAIN CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Log Analyzer & Ticket Classifier")
    parser.add_argument("--train", action="store_true", help="Train the ML model")
    parser.add_argument("--ticket", type=str, help="Predict category for a ticket text")
    parser.add_argument("--logfile", type=str, help="Analyze a log file")

    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.ticket:
        predict_ticket(args.ticket)
    elif args.logfile:
        analyze_log(args.logfile)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

