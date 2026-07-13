from __future__ import annotations

from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


UrlOpener = Callable[..., Any]


def probe_openai_endpoint(
    base_url: str | None,
    *,
    api_key: str | None = None,
    timeout_s: float = 1.5,
    opener: UrlOpener = urlopen,
) -> dict[str, Any]:
    if not base_url:
        return {
            "checked": False,
            "url": None,
            "reachable": None,
            "authorized": None,
            "usable": None,
            "status_code": None,
            "error": None,
        }

    url = f"{base_url.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers)
    try:
        response = opener(request, timeout=timeout_s)
        try:
            status_code = int(response.getcode())
        finally:
            close = getattr(response, "close", None)
            if close:
                close()
        return _endpoint_result(url, status_code=status_code)
    except HTTPError as exc:
        try:
            return _endpoint_result(url, status_code=int(exc.code))
        finally:
            exc.close()
    except (URLError, OSError, TimeoutError, ValueError) as exc:
        return {
            "checked": True,
            "url": url,
            "reachable": False,
            "authorized": None,
            "usable": False,
            "status_code": None,
            "error": type(exc).__name__,
        }


def _endpoint_result(url: str, *, status_code: int) -> dict[str, Any]:
    authorized = status_code not in {401, 403}
    return {
        "checked": True,
        "url": url,
        "reachable": True,
        "authorized": authorized,
        "usable": authorized and status_code < 500,
        "status_code": status_code,
        "error": None,
    }
