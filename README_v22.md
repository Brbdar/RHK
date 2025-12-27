# RHK Befundassistent (DE) – Web-App (v22)

Eine **rein deutschsprachige** Gradio-Web-App zur strukturierten Eingabe von:
- Klinik/Labor (inkl. Red Flags)
- Bildgebung (Echo/CT/MRT)
- Rechtsherzkatheter (Ruhe, Belastung, Volumen, Vasoreagibilität)
- Optional: Zusatzmodule / Textbausteine

und zur **automatischen Ableitung** von:
- Hämodynamik (mPAP, TPG, DPG, PVR, CI …)
- Belastungskennzahlen (mPAP/CO‑Slope, PAWP/CO‑Slope …)
- Echo-Parameter (TAPSE/sPAP inkl. 3‑Strata‑Einordnung, RAAI, S’/RAAI …)
- **Risikostratifizierung** (ESC/ERS 4‑Strata, ESC/ERS 3‑Strata, **ESC/ERS „comprehensive“** 3‑Strata, **REVEAL Lite 2**, H2FPEF)

Außerdem gibt es eine **regelbasierte Empfehlungsebene** (YAML‑Rulebook), die abhängig von eingegebenen/abgeleiteten Parametern:
- Red‑Flags markiert
- fehlende Kernwerte anfordert
- Empfehlungen / Textbausteine vorschlägt
- optionale Zusatzmodule zur Beurteilung ergänzt

> Hinweis: Dieses Projekt ist **kein Medizinprodukt**. Es ersetzt keine ärztliche Beurteilung.

---

## Dateien (Source of Truth)

- `rhk_app_web_master_v22.py`  
  **Die Haupt-App.** Enthält UI, Berechnungen, Risiko-Scores, Berichtsgenerator, Rule-Engine.

- `rhk_rules_v22.yaml`  
  **Regelbuch (Rulebook).** Enthält When/Then-Regeln für Empfehlungen, Red Flags, Pflichtfelder.

- `requirements_v22.txt`  
  Abhängigkeiten für lokalen Betrieb/Render.

---

## Lokaler Start

### Option A: direkt per Python
```bash
pip install -r requirements_v22.txt
python rhk_app_web_master_v22.py
```

### Option B: Jupyter Notebook (Snippet)
```python
import os
os.environ["PORT"] = "7860"           # optional
os.environ["RHK_RULEBOOK"] = "rhk_rules_v22.yaml"  # optional
from rhk_app_web_master_v22 import main
main()
```

---

## Deployment auf Render (Kurzfassung)

1. Repository mit den drei Dateien in Render verbinden.
2. Build Command:
```bash
pip install -r requirements_v22.txt
```
3. Start Command:
```bash
python rhk_app_web_master_v22.py
```

Optional (Render ENV):
- `RHK_RULEBOOK`: Pfad zum YAML‑Rulebook (Default: `rhk_rules_v22.yaml`)
- `PORT`: Render setzt diesen Wert meist automatisch.

---

## Datenmodell (wichtig für jede LLM / Erweiterung)

Die App arbeitet intern mit drei zentralen Dicts:

### 1) `ui` (Rohdaten)
Enthält alle UI‑Inputs **1:1** nach Schlüsselname (z.B. `who_fc`, `rap_rest`, `pasp_echo`, …).

### 2) `derived` (abgeleitete Werte)
Wird aus `ui` berechnet (z.B. `mpap`, `pvr`, `ci`, `tapse_spap`, `tapse_spap_risk`, Slopes …).

### 3) `case`
Ein Gesamtobjekt, das im Output/Report verwendet wird:
- `case["ui"]`
- `case["derived"]`
- `case["scores"]` (Risikokategorien, Punkte, Mittelwerte)
- `case["recommendations"]`, `case["required_fields"]`, `case["modules"]`, `case["tags"]` (aus Rulebook)

**Wichtig:** Das YAML‑Rulebook sieht sowohl `ui`‑Felder als auch `derived`‑Felder (und `scores`) im selben „Environment“.  
Daher müssen neue Parameter **entweder** als UI‑Feld existieren **oder** als `derived`‑Wert berechnet und ins Env geschrieben werden.

---

## Risikoscores (v22)

### ESC/ERS 4‑Strata & 3‑Strata (Follow‑Up)
- Basierend auf WHO‑FC, 6MWD, BNP/NT‑proBNP (vereinfachte Umsetzung, wie in der App bisher)

### ESC/ERS „comprehensive“ 3‑Strata (Table 16 – vereinfacht)
- Bewertet nur verfügbare Parameter (u.a. WHO‑FC, 6MWD, BNP/NT‑proBNP, Synkope, RA‑ESA, Perikarderguss, RAP, CI, SvO2, TAPSE/sPAP, CMR‑RVEF)
- Mapping über Mittelwert der Einzelgrade:
  - <= 1.5 → niedrig
  - 1.5–2.5 → intermediär
  - > 2.5 → hoch

### REVEAL Lite 2
- WHO‑FC, 6MWD, BNP/NT‑proBNP, RRsys, HF, eGFR
- Ausgabe: Kategorie + Punktzahl

### TAPSE/sPAP
- Zusätzlich zum Tello‑Cutoff (`<0.31`) wird eine 3‑Strata‑Einordnung ausgegeben:
  - >0.32 niedrig
  - 0.19–0.32 intermediär
  - <0.19 hoch

---

## Rulebook (YAML) – wie es funktioniert

Datei: `rhk_rules_v22.yaml`

Ein Rule-Eintrag hat das Schema:
```yaml
- id: R_SOME_RULE
  priority: 200
  when: "some_field == True and pvr > 2"
  then:
    add_recommendations:
      - "Text..."
    require_fields:
      - "mpap"
    add_tags:
      - "Red Flag"
    add_modules:
      - "P13"
```

### Wichtige Punkte
- `when` ist ein Python‑ähnlicher Ausdruck (sicher ausgewertet).
- `priority`: höhere Zahl = wird früher angewendet.
- `require_fields`: Schlüssel, die im Dashboard als fehlend erscheinen.
- `add_modules`: IDs von Textbausteinen (in `rhk_app_web_master_v22.py` als `TEXT_BLOCKS` hinterlegt).
- `add_tags`: z.B. „Red Flag“, „Anämie“, …

### Erweiterung (Best Practice)
1. **Neuen Parameter**: UI hinzufügen (build_demo) oder als `derived` berechnen (build_case).
2. **Im Rulebook**: neue Regel mit sauberer Bedingung + klarer Empfehlung.
3. Optional: **Textbaustein** in `TEXT_BLOCKS` ergänzen und per `add_modules` referenzieren.

---

## Abwärtskompatibilität

Beim Laden alter JSON‑Cases wird automatisch gemappt:
- `syncope: true/false` → `"gelegentlich"` / `"keine"`
- `anemia_type: microcytic/normocytic/macrocytic/...` → deutsche Werte

---

## Typische Fehlerquellen

- **Belastungsslope wird nicht berechnet**: Es braucht i.d.R. mPAP (Ruhe & Peak) und CO/CI (Ruhe & Peak).  
  In v22 kann alternativ **CI Peak** eingegeben werden, wenn BSA berechenbar ist.

- **Dropdown-Fehler „Value not in choices“**: meist alte gespeicherte Werte → Case neu speichern oder Mapping ergänzen.

---

## Lizenz / Disclaimer

Dieses Repository ist ein technisches Hilfsmittel. Medizinische Inhalte müssen lokal geprüft und verantwortet werden.
