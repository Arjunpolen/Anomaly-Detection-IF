import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cgi

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"


class RequestHandler(SimpleHTTPRequestHandler):
    def _cors_origin(self):
        allowed = os.getenv("CORS_ORIGIN", "*")
        if allowed == "*":
            return "*"
        return allowed

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", self._cors_origin())
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if self.path != "/api/run":
            self.send_error(404, "Not found")
            return

        content_type = self.headers.get("Content-Type")
        if not content_type:
            self.send_error(400, "Missing Content-Type")
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )

        if "file" not in form:
            self.send_error(400, "file field missing")
            return

        file_item = form["file"]
        filename = Path(file_item.filename or "upload.csv").name
        timestamp = int(time.time())
        upload_path = UPLOAD_DIR / f"{timestamp}_{filename}"

        with upload_path.open("wb") as f:
            f.write(file_item.file.read())

        python_exec = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
        cmd = [
            python_exec,
            str(PROJECT_ROOT / "source" / "run_pipeline.py"),
            "--input",
            str(upload_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "status": "error",
                        "message": result.stderr.strip() or "Pipeline failed",
                    }
                ).encode("utf-8")
            )
            return

        summary_path = PROJECT_ROOT / "results" / "summary.json"
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())

        cache_bust = f"?t={timestamp}"
        plots = {
            "confusion_matrix": f"/results/plots/confusion_matrix.png{cache_bust}",
            "roc_curve": f"/results/plots/roc_curve.png{cache_bust}",
            "score_distribution": f"/results/plots/score_distribution.png{cache_bust}",
            "metrics_snapshot": f"/results/plots/metrics_snapshot.png{cache_bust}",
            "dashboard": f"/results/plots/dashboard.png{cache_bust}",
        }

        response = {
            "status": "ok",
            "summary": summary,
            "plots": plots,
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))


def _create_server(port):
    return ThreadingHTTPServer(("", port), RequestHandler)


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="cgi")
    parser = argparse.ArgumentParser(description="Run local upload server.")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()

    start_port = args.port
    httpd = None
    bound_port = None

    for port in range(start_port, start_port + 10):
        try:
            httpd = _create_server(port)
            bound_port = port
            break
        except OSError as exc:
            if exc.errno in {48, 98}:  # address already in use
                continue
            raise

    if httpd is None:
        raise OSError(f"No available port found from {start_port} to {start_port + 9}.")

    print(f"Serving on http://localhost:{bound_port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
