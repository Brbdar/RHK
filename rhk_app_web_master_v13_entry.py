# -*- coding: utf-8 -*-
"""
Compatibility wrapper.

If your Render start command still points to:
  python rhk_app_web_master_v13.py

…then replace that file in your repo with this content OR change the start command
to use rhk_app_web_master_v15.py.

This wrapper simply exposes the FastAPI app from v15.
"""
from rhk_app_web_master_v15 import app  # noqa: F401

if __name__ == "__main__":
    import rhk_app_web_master_v15  # noqa: F401
