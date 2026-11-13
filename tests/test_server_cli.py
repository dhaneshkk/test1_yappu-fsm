import json
import threading
import urllib.request

from conftest import POEM_2

from yappu_fsm.cli import main
from yappu_fsm.server import serve


def test_openai_compatible_endpoint():
    srv = serve(port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        def post(body):
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req).read())

        r = post({"model": "yappu-icd-linker", "messages": [{"role": "user", "content": "Denies SOB, but has HTN."}]})
        assert r["object"] == "chat.completion"
        content = json.loads(r["choices"][0]["message"]["content"])
        assert content["allowed_codes"] == ["I10"]

        r = post({"model": "yappu-venba-validator", "messages": [{"role": "user", "content": POEM_2}]})
        assert "VALID Nerisai Venba" in r["choices"][0]["message"]["content"]

        r = post({"model": "yappu-icd-rules", "setting": "outpatient",
                  "messages": [{"role": "user", "content": "Pt with T2DM and CKD. Also HTN."}]})
        assert json.loads(r["choices"][0]["message"]["content"])["codes"] == ["E11.22", "I12.9", "N18.9"]

        models = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models").read())
        assert {m["id"] for m in models["data"]} == {"yappu-icd-linker", "yappu-icd-rules", "yappu-venba-validator"}
    finally:
        srv.shutdown()


def test_cli(capsys):
    assert main(["asai", "கூடினால்"]) == 0
    assert "Koovilam" in capsys.readouterr().out
    assert main(["validate", POEM_2]) == 0
    assert main(["validate", "எண்ணும் வானில்"]) == 1
    main(["code", "Pt has lt knee osteoarthritis"])
    assert "M17.12" in capsys.readouterr().out
    main(["code", "Pt with T2DM and CKD.", "--rules"])
    assert "E11.22" in capsys.readouterr().out
    main(["rules"])
    assert "rules OK" in capsys.readouterr().out
    main(["rules", "--datalog"])
    assert "final(P, C)" in capsys.readouterr().out
    main(["eval"])
    assert "assertion_mcc" in capsys.readouterr().out
