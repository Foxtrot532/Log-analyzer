# Log Analyzer & Ticket Classifier (Hybrid AI Tool)

**Author:** Lakshmi Ajithesh  
**Tech Stack:** Python, Regex, scikit-learn, pandas, joblib  

---

## 🏆 Overview
This project combines **regex-based log analysis** with **ML-based support ticket classification** to automate root-cause identification and recurring error pattern detection.  

It is designed to help support engineers quickly identify issues from logs and categorize tickets efficiently, showcasing both **automation skills** and **analytical thinking**.

---

## ⚙️ Features
- **Regex Log Analyzer:**  
  Scans system logs for keywords like `ERROR`, `FAIL`, `TIMEOUT`, `DISK`, `NETWORK`, `CRASH` and suggests the probable issue category.

- **ML Ticket Classifier:**  
  Uses `TfidfVectorizer` + `LogisticRegression` to classify support tickets into categories:  
  `Software Bug`, `Network Issue`, `Storage Issue`, `Authentication Issue`, `Hardware Fault`.

- **Combined CLI Interface:**  
  Analyze logs and classify tickets in one command.

---

## 📁 Project Structure

log_analyzer_ticket_classifier/
│
├── analyzer.py # main Python script (regex + ML)
├── tickets.csv # mock support ticket dataset
├── logs.txt # sample log file
├── README.md
└── LICENSE


---

## 🚀 Installation
1. Clone the repo:
```bash
git clone <your-repo-url>
cd log_analyzer_ticket_classifier

    Install dependencies:

pip install -r requirements.txt

🏃 Usage
1. Train the ML model:

python analyzer.py --train

2. Analyze a log file:

python analyzer.py --logfile logs.txt

3. Classify a ticket:

python analyzer.py --ticket "Disk space full on server"

4. Combine log analysis & ticket classification:

python analyzer.py --logfile logs.txt --ticket "VPN connection timeout"

Sample Output:

--- Log Analysis Summary ---
ERROR       : 3 occurrence(s)
NETWORK     : 1 occurrence(s)
CRASH       : 1 occurrence(s)

[+] Probable Log Category: Software Bug
[+] Predicted Ticket Category: Network Issue

📈 Future Improvements

    Add Tkinter GUI for interactive usage.

    Integrate real-time log monitoring.

    Expand ML dataset for better accuracy.

    Use advanced NLP models for ticket classification.
