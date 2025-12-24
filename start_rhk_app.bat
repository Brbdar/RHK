@echo off
echo ============================================
echo  RHK Befundgenerator – Start (Windows)
echo ============================================

REM Wechsle ins Verzeichnis der .bat-Datei
cd /d %~dp0

REM Prüfe, ob virtuelle Umgebung existiert
IF NOT EXIST .venv (
    echo [INFO] Virtuelle Umgebung nicht gefunden.
    echo [INFO] Erstelle virtuelle Umgebung...
    python -m venv .venv
)

REM Aktiviere virtuelle Umgebung
echo [INFO] Aktiviere virtuelle Umgebung...
call .venv\Scripts\activate

REM Pip aktualisieren
echo [INFO] Aktualisiere pip...
pip install --upgrade pip

REM Abhaengigkeiten installieren
echo [INFO] Installiere Abhaengigkeiten...
pip install -r requirements.txt

REM Starte die Web-App
echo [INFO] Starte RHK Web-App...
python rhk_app_web.py

pause
