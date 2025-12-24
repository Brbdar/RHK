# RHK Befundgenerator – Web‑GUI + Textbaustein‑Datenbank (YAML)

Dieses Paket bündelt zwei Dinge:

1) **Web‑GUI (Gradio)** zur Dateneingabe & Befundgenerierung: `rhk_app_web.py`
2) **Textbaustein‑Datenbank (YAML‑basiert)** inkl. Engine (Add‑ons/Methodik‑Hinweise) und Override‑Workflow:
   - `textdb/core.yaml` (Release / read‑only)
   - `textdb/overrides.yaml` (lokale Änderungen, draft/approved)
   - `rhk_textdb_api.py`, `textdb_store.py`, `rhk_textdb_engine.py`

Die GUI generiert einen strukturierten Befundtext, nutzt dabei die Bausteine aus der Datenbank
und kann (optional) die Datenbank **direkt in der GUI** über Overrides erweitern.

---

## Schnellstart

### 1) Voraussetzungen

- Python **3.10+** empfohlen
- Keine Internetverbindung erforderlich (lokal)

### 2) Installation

Im Paketordner:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 3) Start

```bash
python rhk_app_web.py
```

Dann im Browser die lokale Gradio‑Adresse öffnen (wird im Terminal angezeigt).

---

## Ordnerstruktur

```
.
├── rhk_app_web.py                 # Web‑GUI (Gradio)
├── rhk_textdb_api.py              # API/Loader + Override‑Wrapper + Kompatibilität
├── rhk_textdb_engine.py           # Engine: Ableitungen, Klassifikation, Add‑ons
├── textdb_store.py                # YAML Store, Merge core+overrides, TextBlock‑Klasse
├── validate_textdb.py             # Validator (core.yaml)
├── requirements.txt
└── textdb/
    ├── core.yaml                  # RELEASE‑Datenbank (nicht lokal editieren)
    ├── overrides.yaml             # Lokale Overrides (hier editieren oder per GUI)
    └── references.yaml            # Literatur/Link‑Ablage (optional)
```

---

## Was wurde in der GUI technisch umgebaut?

### 1) TextDB‑Loader

Die GUI lädt **bevorzugt** `rhk_textdb_api.py` (YAML‑backed). Dadurch stehen zur Verfügung:

- `textdb.get_block(id)`
- `textdb.ALL_BLOCKS`, `textdb.BUNDLES`, `textdb.QUICKFINDER`
- **Kompatibilität für GUI**: `textdb.P_BLOCKS` und `textdb.DEFAULT_RULES`
- Engine‑Funktionen: `textdb.suggest_plan(...)`, `textdb.build_context(...)`, ...

### 2) Neuer Abschnitt im Befund: “METHODIK / QUALITÄT (Auto)”

Im generierten Befund wird zwischen “HÄMODYNAMIK” und “KLINIK / LABORE” ein zusätzlicher
Abschnitt eingefügt:

- automatisch erzeugte **Methodik‑/Qualitätshinweise** (Add‑on‑Blöcke `BZ..`), z.B.
  - PAWP‑Grauzone / Messunsicherheit (optional)
  - Stufenoxymetrie‑Hinweise (bei Sprung)
  - zusätzliche Hämodynamik‑Ableitungen (TPG/DPG/PAC/SVI/PAPI)
  - simple hämodynamische Risiko‑Marker (RAP/CI/SvO2/SVI‑basiert)

### 3) Neue optionale Eingaben (Qualität / PAWP)

In Tab **“2) RHK Basis (Ruhe)”** gibt es das Akkordeon:

**“Messqualität / PAWP‑Validierung (optional)”**

- Wedge‑Sättigung (%) (falls erhoben)
- Checkboxen (falls relevant):
  - große respiratorische Schwankungen
  - Adipositas / erhöhter pleuraler Druck
  - COPD / Auto‑PEEP
  - Beatmung / PEEP

Diese Angaben werden in `additional_measurements.quality` abgelegt und von der Engine genutzt.

### 4) TextDB‑Editor direkt in der GUI

Neuer Tab: **“9) TextDB / Bausteine”**

- Block auswählen, bearbeiten, als **Draft** speichern oder **Approve**
- Neuer Block kann durch Eingabe einer neuen ID erstellt werden
- Änderungen landen in: `textdb/overrides.yaml`
- “TextDB neu laden” aktualisiert die Datenbank **ohne Neustart** der GUI

---

## Datenbank‑Prinzip

### core.yaml (Release)

- Enthält die “offizielle” Bausteinsammlung inkl.
  - Blöcke (Katalog, Prozeduren, Boards, Add‑ons)
  - Bundles (z.B. `K05` → `K05_B`, `K05_E`)
  - Regeln (`rules`) für Engine‑Hinweise

**Nicht** im Klinikalltag editieren. Änderungen hier sind “Release‑Änderungen”.

### overrides.yaml (lokal)

- Dient für lokale Anpassungen
- Enthält `draft` und `approved`

Workflow:

1. Änderung/Neuer Block als **draft** anlegen
2. medizinisch prüfen
3. **approve** → wird “wirksam” (überschreibt core)

Die GUI bietet dafür Buttons.

---

## Textblöcke – Felder & Best Practices

Ein Block besteht im Kern aus:

- `id`: z.B. `P12`, `BZ17_ADVANCED_DERIVATIONS`, `K05_B`
- `title`: lesbarer Titel
- `category`: z.B. `P`, `K_B`, `K_E`, `BZ`, `BE` …
- `kind`: z.B. `assessment`, `recommendation`, `procedure_recommendation`, `addon_detail`
- `template`: Text mit Platzhaltern `{...}`
- optional: `tags`, `applies_to`, `priority`, `variants`, `notes`

### Platzhalter (Template‑Variablen)

Platzhalter werden beim Generieren aus einem Kontext befüllt. Nicht vorhandene Keys erzeugen
**keinen Crash** (werden leer), daher gilt:

- neue Platzhalter **immer** einmal im GUI‑Output “JSON (Debug)” prüfen
- oder im internen Output kontrollieren

Praktischer Tipp:

1) Befund generieren
2) rechts unten “JSON (Debug)” öffnen
3) dort sieht man alle Key‑Namen, die Templates verwenden können

---

## Regeln / Cutoffs

Die GUI hat unter **“8) Advanced”** ein Set an Cutoffs.

- Für die **GUI‑Klassifikation** werden diese Werte direkt genutzt.
- Für die **Engine‑Add‑ons** werden diese Werte in das neue Rules‑Schema gemappt
  (siehe `_patch_engine_rules(...)`).

Damit bleiben die Auto‑Hinweise konsistent, wenn du Cutoffs änderst.

---

## Validierung

Validator für `core.yaml`:

```bash
python validate_textdb.py --core textdb/core.yaml
```

Hinweis: Der Validator prüft u.a. Template‑Konsistenz (Platzhalter innerhalb eines Blocks).
Er prüft **nicht**, ob ein Platzhalter im GUI‑Kontext existiert – dafür ist “JSON (Debug)” die Referenz.

---

## Troubleshooting

### “Keine Textbaustein‑Datenbank gefunden”

Stelle sicher, dass **`rhk_textdb_api.py`** und der Ordner `textdb/` im selben Ordner liegen wie
`rhk_app_web.py`.

### Änderungen am Text erscheinen nicht

- Im Tab “TextDB / Bausteine” **TextDB neu laden** klicken
- oder GUI neu starten

### Ich will nur lokal am Text drehen (ohne core.yaml anzufassen)

- ändere im GUI‑Tab “TextDB / Bausteine” und speichere als Draft/Approved
- oder editiere `textdb/overrides.yaml` direkt

---

## Nächste Schritte (für die nächste Runde GUI‑Feinschliff)

Wenn wir den GUI‑Workflow noch “perfekter” machen sollen, sind typische Erweiterungen:

- “Empfohlene Bausteine” live anzeigen (Auto‑Vorschläge als Checkbox‑Liste)
- per Klick Add‑ons ein/ausblenden
- Quickfinder‑Suche in der GUI (Tags/Symptome → passende Blocks)
- Versionierung/Changelog pro Override‑Änderung

