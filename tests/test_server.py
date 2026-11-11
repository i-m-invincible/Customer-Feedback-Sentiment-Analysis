"""Exercise the actual HTTP server as well as the AppTest interaction tests."""

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.request import urlopen

from sentiment import ROOT


class ServerTests(unittest.TestCase):
    def test_streamlit_serves_a_healthy_app(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
            server = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app.py",
                 "--server.headless=true", "--server.address=127.0.0.1",
                 f"--server.port={port}", "--browser.gatherUsageStats=false"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline and server.poll() is None:
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                            if response.status == 200 and response.read() == b"ok":
                                break
                    except (URLError, TimeoutError):
                        time.sleep(0.25)
                else:
                    log.seek(0)
                    self.fail("Streamlit did not become healthy:\n" + log.read())
                with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(b"<html", response.read().lower())
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
