from rhk.migrate import migrate_payload_to_ui, build_saved_case


def test_migrate_new_format():
    payload = {"schema":"rhk_case","schema_version":3,"ui":{"a":1}}
    ui, msg = migrate_payload_to_ui(payload)
    assert ui["a"] == 1
    assert "Schema" in msg or "geladen" in msg


def test_migrate_legacy_flat():
    payload = {"last_name":"X","mpap":30}
    ui, msg = migrate_payload_to_ui(payload)
    assert ui["mpap"] == 30


def test_build_saved_case():
    sc = build_saved_case({"x":1})
    assert sc["schema"] == "rhk_case"
    assert "ui" in sc and sc["ui"]["x"] == 1
