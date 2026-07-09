from __future__ import annotations

import unittest
from urllib.error import HTTPError, URLError

from gem_rags.endpoint import probe_openai_endpoint


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.closed = False

    def getcode(self) -> int:
        return self.status_code

    def close(self) -> None:
        self.closed = True


class TestEndpoint(unittest.TestCase):
    def test_probe_reports_reachable_authorized_endpoint(self) -> None:
        response = _Response(200)

        report = probe_openai_endpoint(
            "http://localhost:8000/v1/",
            api_key="secret",
            opener=lambda request, timeout: response,
        )

        self.assertEqual(report["url"], "http://localhost:8000/v1/models")
        self.assertTrue(report["reachable"])
        self.assertTrue(report["authorized"])
        self.assertTrue(report["usable"])
        self.assertTrue(response.closed)
        self.assertNotIn("secret", str(report))

    def test_probe_distinguishes_auth_and_connection_failures(self) -> None:
        def unauthorized(request, timeout):
            raise HTTPError(request.full_url, 401, "unauthorized", {}, None)

        def unavailable(request, timeout):
            raise URLError("connection refused")

        auth = probe_openai_endpoint("http://localhost:8000/v1", opener=unauthorized)
        down = probe_openai_endpoint("http://localhost:8000/v1", opener=unavailable)

        self.assertTrue(auth["reachable"])
        self.assertFalse(auth["authorized"])
        self.assertFalse(auth["usable"])
        self.assertFalse(down["reachable"])
        self.assertIsNone(down["authorized"])
        self.assertEqual(down["error"], "URLError")

    def test_probe_skips_missing_base_url(self) -> None:
        report = probe_openai_endpoint(None)

        self.assertFalse(report["checked"])
        self.assertIsNone(report["usable"])


if __name__ == "__main__":
    unittest.main()
