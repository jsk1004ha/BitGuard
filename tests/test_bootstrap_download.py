from __future__ import annotations

import hashlib
import json
import os
import socket
import tempfile
import threading
import unittest
from contextlib import contextmanager
from dataclasses import replace
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from bitguard_bnn.bootstrap import download as download_module
from bitguard_bnn.bootstrap import manifest as manifest_module
from bitguard_bnn.bootstrap.download import DownloadError, download_file, sanitize_url
from bitguard_bnn.bootstrap.manifest import (
    SourceManifestError,
    build_source_manifest,
    manifest_json_bytes,
    write_source_manifest,
)
from bitguard_bnn.bootstrap.registry import load_registry


PAYLOAD = bytes(range(251)) * 17
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class _RangeServer:
    def __init__(self, data: bytes = PAYLOAD, *, mode: str = "range") -> None:
        self.data = data
        self.mode = mode
        self.requests: list[dict[str, str]] = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                fixture.requests.append({key: value for key, value in self.headers.items()})
                requested = self.headers.get("Range")
                if fixture.mode in {"redirect-secret", "redirect-error"} and self.path.startswith(
                    "/archive.zip"
                ):
                    self.send_response(302)
                    self.send_header(
                        "Location",
                        "/redirected.zip?access_token=redirect-query-secret"
                        "#redirect-fragment-secret",
                    )
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if fixture.mode == "redirect-error" and self.path.startswith(
                    "/redirected.zip"
                ):
                    self.send_error(503, "redirected fixture failure")
                    return
                if fixture.mode == "error":
                    self.send_error(503, "temporary fixture failure")
                    return

                start = None
                if requested and requested.startswith("bytes=") and requested.endswith("-"):
                    try:
                        start = int(requested[6:-1])
                    except ValueError:
                        start = None

                if fixture.mode == "range-416" and start is not None:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(fixture.data)}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                if start is not None and fixture.mode != "ignore-range":
                    body = fixture.data[start:]
                    self.send_response(206)
                    if fixture.mode == "invalid-range":
                        content_start = start + 1
                    else:
                        content_start = start
                    end = content_start + len(body) - 1
                    self.send_header(
                        "Content-Range",
                        f"bytes {content_start}-{end}/{len(fixture.data)}",
                    )
                else:
                    body = fixture.data
                    self.send_response(200)

                declared_length = len(body)
                if fixture.mode == "short-body":
                    declared_length += 19
                self.send_header("Content-Length", str(declared_length))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                if fixture.mode == "short-body":
                    self.close_connection = True

            def log_message(self, _format: str, *args: object) -> None:
                del args

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        host, port = self.httpd.server_address
        self.url = f"http://{host}:{port}/archive.zip"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> _RangeServer:
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise AssertionError("range test server thread did not stop")


class _GuardedResponse:
    def __init__(self, response: object, reads: list[int]) -> None:
        self._response = response
        self._reads = reads

    def read(self, amount: int = -1) -> bytes:
        if not isinstance(amount, int) or amount <= 0:
            raise AssertionError("response reads must use a fixed positive integer chunk")
        self._reads.append(amount)
        return self._response.read(amount)  # type: ignore[attr-defined]

    def __enter__(self) -> _GuardedResponse:
        self._response.__enter__()  # type: ignore[attr-defined]
        return self

    def __exit__(self, *args: object) -> object:
        return self._response.__exit__(*args)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self._response, name)


class _CloseTrackingHTTPError(HTTPError):
    def __init__(self, url: str, status: int, headers: Message | None = None) -> None:
        super().__init__(url, status, "fixture error", headers or Message(), BytesIO())
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        super().close()


class _InterruptingResponse:
    def __init__(self, url: str, signal: BaseException, prefix: bytes) -> None:
        self.status = 200
        self.headers = Message()
        self.headers["Content-Length"] = str(len(PAYLOAD))
        self._url = url
        self._signal = signal
        self._prefix = prefix
        self._reads = 0
        self.close_count = 0

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return self.status

    def read(self, _amount: int) -> bytes:
        self._reads += 1
        if self._reads == 1:
            return self._prefix
        raise self._signal

    def close(self) -> None:
        self.close_count += 1

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class _CloseTrackingStream:
    def __init__(self, stream: object) -> None:
        self._stream = stream
        self.close_count = 0

    def write(self, payload: bytes) -> int:
        return self._stream.write(payload)  # type: ignore[attr-defined]

    def flush(self) -> None:
        self._stream.flush()  # type: ignore[attr-defined]

    def fileno(self) -> int:
        return self._stream.fileno()  # type: ignore[attr-defined]

    def close(self) -> None:
        self.close_count += 1
        self._stream.close()  # type: ignore[attr-defined]


class DownloadFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / "archive.zip"
        self.partial = self.root / "archive.zip.partial"

    def test_half_partial_resumes_without_duplicate_append(self) -> None:
        half = len(PAYLOAD) // 2
        self.partial.write_bytes(PAYLOAD[:half])
        with _RangeServer() as server:
            result = download_file(
                server.url,
                self.destination,
                expected_sha256=PAYLOAD_SHA256,
                chunk_size=97,
            )

        self.assertEqual(server.requests[0]["Range"], f"bytes={half}-")
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertFalse(self.partial.exists())
        self.assertEqual(result.sha256, PAYLOAD_SHA256)
        self.assertEqual(result.byte_size, len(PAYLOAD))
        self.assertTrue(result.resumed)
        self.assertFalse(result.restarted)
        self.assertFalse(result.reused)
        self.assertEqual(json.loads(json.dumps(result.to_dict())), result.to_dict())

    def test_ignored_range_restarts_from_zero_and_truncates_partial(self) -> None:
        half = len(PAYLOAD) // 2
        self.partial.write_bytes(PAYLOAD[:half])
        with _RangeServer(mode="ignore-range") as server:
            result = download_file(server.url, self.destination, chunk_size=101)

        self.assertEqual(server.requests[0]["Range"], f"bytes={half}-")
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertEqual(result.sha256, PAYLOAD_SHA256)
        self.assertFalse(result.resumed)
        self.assertTrue(result.restarted)

    def test_restart_validates_opened_inode_before_truncating(self) -> None:
        prefix = PAYLOAD[:79]
        winner = b"foreign replacement must remain byte-identical"
        displaced = self.root / "displaced-original.partial"
        replacement = self.root / "foreign-replacement.partial"
        self.partial.write_bytes(prefix)
        replacement.write_bytes(winner)
        real_open = download_module.os.open
        substituted = False

        def substitute_before_open(path: object, flags: int, mode: int = 0o777) -> int:
            nonlocal substituted
            if (
                not substituted
                and Path(path) == self.partial
                and flags & os.O_WRONLY
            ):
                substituted = True
                self.partial.replace(displaced)
                replacement.replace(self.partial)
            return real_open(path, flags, mode)

        with _RangeServer(mode="ignore-range") as server:
            with patch.object(download_module.os, "open", side_effect=substitute_before_open):
                with self.assertRaisesRegex(DownloadError, "changed|identity"):
                    download_file(server.url, self.destination)

        self.assertTrue(substituted)
        self.assertEqual(self.partial.read_bytes(), winner)
        self.assertEqual(displaced.read_bytes(), prefix)
        self.assertFalse(self.destination.exists())

    def test_restart_does_not_follow_a_partial_symlink_substitution(self) -> None:
        prefix = PAYLOAD[:67]
        winner = b"outside file must never be truncated"
        outside = self.root / "outside-restart-target"
        probe = self.root / "symlink-privilege-probe"
        outside.write_bytes(winner)
        try:
            probe.symlink_to(outside)
            probe.unlink()
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")

        displaced = self.root / "displaced-symlink-original.partial"
        self.partial.write_bytes(prefix)
        real_open = download_module.os.open
        substituted = False

        def substitute_symlink_before_open(
            path: object, flags: int, mode: int = 0o777
        ) -> int:
            nonlocal substituted
            if (
                not substituted
                and Path(path) == self.partial
                and flags & os.O_WRONLY
            ):
                substituted = True
                self.partial.replace(displaced)
                self.partial.symlink_to(outside)
            return real_open(path, flags, mode)

        with _RangeServer(mode="ignore-range") as server:
            with patch.object(
                download_module.os,
                "open",
                side_effect=substitute_symlink_before_open,
            ):
                with self.assertRaises(DownloadError):
                    download_file(server.url, self.destination)

        self.assertTrue(substituted)
        self.assertEqual(outside.read_bytes(), winner)
        self.assertEqual(displaced.read_bytes(), prefix)
        self.assertFalse(self.destination.exists())

    def test_invalid_content_range_never_appends_or_publishes(self) -> None:
        prefix = PAYLOAD[:53]
        self.partial.write_bytes(prefix)
        with _RangeServer(mode="invalid-range") as server:
            with self.assertRaisesRegex(DownloadError, "Content-Range"):
                download_file(server.url, self.destination)

        self.assertEqual(self.partial.read_bytes(), prefix)
        self.assertFalse(self.destination.exists())

    def test_short_body_keeps_actionable_partial_without_final(self) -> None:
        prefix = PAYLOAD[:73]
        self.partial.write_bytes(prefix)
        with _RangeServer(mode="short-body") as server:
            with self.assertRaisesRegex(DownloadError, "length|response|download"):
                download_file(server.url, self.destination, chunk_size=83)

        self.assertTrue(self.partial.exists())
        self.assertTrue(self.partial.read_bytes().startswith(prefix))
        self.assertFalse(self.destination.exists())

    def test_http_and_network_failures_preserve_existing_partial(self) -> None:
        prefix = PAYLOAD[:41]
        for failure in ("http", "network"):
            with self.subTest(failure=failure):
                self.partial.write_bytes(prefix)
                if failure == "http":
                    fixture = _RangeServer(mode="error")
                    context = fixture
                else:
                    listener = socket.socket()
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                    listener.close()
                    fixture = None
                    context = _null_context()
                with context:
                    url = fixture.url if fixture is not None else f"http://127.0.0.1:{port}/x"
                    with self.assertRaises(DownloadError) as caught:
                        download_file(url, self.destination, timeout=0.25)
                self.assertNotIn("password", str(caught.exception).lower())
                self.assertEqual(self.partial.read_bytes(), prefix)
                self.assertFalse(self.destination.exists())

    def test_already_verified_final_is_reused_without_network(self) -> None:
        self.destination.write_bytes(PAYLOAD)
        result = download_file(
            "http://127.0.0.1:1/never-requested",
            self.destination,
            expected_sha256=PAYLOAD_SHA256,
        )

        self.assertTrue(result.reused)
        self.assertEqual(result.sha256, PAYLOAD_SHA256)
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_existing_final_without_expected_hash_is_not_silently_trusted(self) -> None:
        self.destination.write_bytes(PAYLOAD)
        with self.assertRaisesRegex(DownloadError, "expected.*hash|persisted"):
            download_file("http://127.0.0.1:1/not-requested", self.destination)

        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_hash_mismatch_remains_partial_and_is_not_published(self) -> None:
        with _RangeServer() as server:
            with self.assertRaisesRegex(DownloadError, "SHA-256"):
                download_file(server.url, self.destination, expected_sha256="0" * 64)

        self.assertFalse(self.destination.exists())
        self.assertEqual(self.partial.read_bytes(), PAYLOAD)

    def test_boolean_timeout_and_chunk_size_are_rejected(self) -> None:
        for timeout in (True, False):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(DownloadError, "timeout"):
                    download_file(
                        "https://example.test/archive",
                        self.destination,
                        timeout=timeout,
                    )

        for chunk_size in (True, False):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaisesRegex(DownloadError, "chunk_size"):
                    download_file(
                        "https://example.test/archive",
                        self.destination,
                        chunk_size=chunk_size,
                    )

    def test_valid_timeout_and_chunk_size_boundaries_are_preserved(self) -> None:
        with _RangeServer() as server:
            result = download_file(
                server.url,
                self.destination,
                timeout=download_module.MAX_TIMEOUT_SECONDS,
                chunk_size=1,
            )

        self.assertEqual(result.sha256, PAYLOAD_SHA256)
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_non_string_expected_sha256_is_rejected_as_download_error(self) -> None:
        for expected_sha256 in (True, False, 1, b"0" * 64, object()):
            with self.subTest(expected_sha256=type(expected_sha256).__name__):
                with self.assertRaisesRegex(DownloadError, "expected_sha256"):
                    download_file(
                        "https://example.test/archive",
                        self.destination,
                        expected_sha256=expected_sha256,  # type: ignore[arg-type]
                    )

    def test_sha256_normalizer_rejects_every_non_string_type(self) -> None:
        for value in (None, True, False, 1, b"0" * 64, object()):
            with self.subTest(value=type(value).__name__):
                with self.assertRaisesRegex(DownloadError, "expected_sha256"):
                    download_module._normalize_sha256(value)

    def test_streams_fixed_chunks_and_fsyncs_before_publication(self) -> None:
        reads: list[int] = []
        fsynced = False
        real_urlopen = download_module.urlopen
        real_fsync = download_module.os.fsync
        real_link = download_module.os.link

        def guarded_urlopen(*args: object, **kwargs: object) -> _GuardedResponse:
            return _GuardedResponse(real_urlopen(*args, **kwargs), reads)

        def record_fsync(descriptor: int) -> None:
            nonlocal fsynced
            fsynced = True
            real_fsync(descriptor)

        def assert_fsynced_before_link(source: object, target: object) -> None:
            self.assertTrue(fsynced)
            real_link(source, target)

        with _RangeServer() as server:
            with (
                patch.object(download_module, "urlopen", side_effect=guarded_urlopen),
                patch.object(download_module.os, "fsync", side_effect=record_fsync),
                patch.object(download_module.os, "link", side_effect=assert_fsynced_before_link),
            ):
                download_file(server.url, self.destination, chunk_size=64)

        self.assertGreater(len(reads), 1)
        self.assertEqual(set(reads), {64})
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_path_substitution_after_streaming_is_not_hashed_or_published(self) -> None:
        winner = b"F" * len(PAYLOAD)
        replacement = self.root / "post-stream-replacement"
        displaced = self.root / "completed-owned-partial"
        replacement.write_bytes(winner)
        real_stream_response = download_module._stream_response

        def substitute_after_stream(*args: object, **kwargs: object):
            result = real_stream_response(*args, **kwargs)
            self.partial.replace(displaced)
            replacement.replace(self.partial)
            return result

        with _RangeServer() as server:
            with patch.object(
                download_module,
                "_stream_response",
                side_effect=substitute_after_stream,
            ):
                with self.assertRaisesRegex(DownloadError, "changed|identity"):
                    download_file(server.url, self.destination)

        self.assertEqual(self.partial.read_bytes(), winner)
        self.assertEqual(displaced.read_bytes(), PAYLOAD)
        self.assertFalse(self.destination.exists())

    def test_concurrent_final_is_never_overwritten(self) -> None:
        winner = b"other verified writer"
        real_link = download_module.os.link

        def concurrent_publish(source: object, target: object) -> None:
            if Path(target) == self.destination:
                Path(target).write_bytes(winner)
                raise FileExistsError(str(target))
            real_link(source, target)

        with _RangeServer() as server:
            with patch.object(download_module.os, "link", side_effect=concurrent_publish):
                with self.assertRaisesRegex(DownloadError, "already exists|concurrent"):
                    download_file(server.url, self.destination)

        self.assertEqual(self.destination.read_bytes(), winner)
        self.assertEqual(self.partial.read_bytes(), PAYLOAD)

    def test_publication_never_links_the_mutable_partial_inode(self) -> None:
        winner = b"P" * len(PAYLOAD)
        replacement = self.root / "publish-replacement"
        displaced = self.root / "verified-owned-partial"
        replacement.write_bytes(winner)
        real_link = download_module.os.link
        substituted = False

        def substitute_before_link(source: object, target: object) -> None:
            nonlocal substituted
            if not substituted and Path(source) == self.partial:
                substituted = True
                self.partial.replace(displaced)
                replacement.replace(self.partial)
            real_link(source, target)

        with _RangeServer() as server:
            with patch.object(download_module.os, "link", side_effect=substitute_before_link):
                result = download_file(server.url, self.destination)

        self.assertFalse(substituted)
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertEqual(result.sha256, PAYLOAD_SHA256)
        self.assertEqual(replacement.read_bytes(), winner)
        self.assertFalse(displaced.exists())
        self.assertFalse(self.partial.exists())

    def test_same_partial_inode_mutated_after_verification_cannot_change_final(self) -> None:
        real_require_identity = download_module._require_path_identity
        mutated = False

        def mutate_after_verified_check(
            path: Path,
            expected: tuple[int, int],
            *,
            subject: str,
        ):
            nonlocal mutated
            result = real_require_identity(path, expected, subject=subject)
            if subject == "Hashed partial" and not mutated:
                mutated = True
                with path.open("r+b", buffering=0) as stream:
                    stream.seek(0)
                    stream.write(b"X" * len(PAYLOAD))
                    stream.flush()
                    os.fsync(stream.fileno())
            return result

        with _RangeServer() as server:
            with patch.object(
                download_module,
                "_require_path_identity",
                side_effect=mutate_after_verified_check,
            ):
                result = download_file(
                    server.url,
                    self.destination,
                    expected_sha256=PAYLOAD_SHA256,
                )

        self.assertTrue(mutated)
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)
        self.assertEqual(result.byte_size, len(PAYLOAD))
        self.assertEqual(result.sha256, hashlib.sha256(self.destination.read_bytes()).hexdigest())

    def test_same_partial_inode_mutated_during_snapshot_is_rejected(self) -> None:
        real_read = download_module.os.read
        partial_identity: tuple[int, int] | None = None
        mutated = False

        def mutate_after_first_snapshot_read(descriptor: int, amount: int) -> bytes:
            nonlocal mutated
            payload = real_read(descriptor, amount)
            descriptor_stat = os.fstat(descriptor)
            descriptor_identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
            if payload and descriptor_identity == partial_identity and not mutated:
                mutated = True
                with self.partial.open("r+b", buffering=0) as stream:
                    stream.seek(0)
                    stream.write(b"Y" * len(PAYLOAD))
                    stream.flush()
                    os.fsync(stream.fileno())
            return payload

        self.partial.write_bytes(PAYLOAD)
        partial_stat = self.partial.stat()
        partial_identity = (partial_stat.st_dev, partial_stat.st_ino)
        with _RangeServer(mode="range-416") as server:
            with patch.object(
                download_module.os,
                "read",
                side_effect=mutate_after_first_snapshot_read,
            ):
                with self.assertRaisesRegex(DownloadError, "changed|snapshot|verified"):
                    download_file(
                        server.url,
                        self.destination,
                        expected_sha256=PAYLOAD_SHA256,
                        chunk_size=64,
                    )

        self.assertTrue(mutated)
        self.assertTrue(self.partial.exists())
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob(".*.verified*")), [])

    def test_substituted_private_download_pin_is_preserved_and_never_reported_as_published(
        self,
    ) -> None:
        real_link = download_module.os.link
        displaced = self.root / "verified-pin-before-swap"
        attacker = b"attacker-controlled-final"
        substituted = False

        def substitute_private_pin(source: object, target: object) -> None:
            nonlocal substituted
            source_path = Path(source)
            target_path = Path(target)
            if (
                not substituted
                and source_path.name.endswith(".verified.pin")
                and target_path == self.destination
            ):
                substituted = True
                source_path.replace(displaced)
                source_path.write_bytes(attacker)
            real_link(source, target)

        with _RangeServer() as server:
            with patch.object(download_module.os, "link", side_effect=substitute_private_pin):
                with self.assertRaisesRegex(DownloadError, "identity|publish|verified"):
                    download_file(
                        server.url,
                        self.destination,
                        expected_sha256=PAYLOAD_SHA256,
                    )

        self.assertTrue(substituted)
        self.assertEqual(self.destination.read_bytes(), attacker)
        self.assertEqual(self.partial.read_bytes(), PAYLOAD)

    def test_foreign_download_replacement_after_link_is_preserved(self) -> None:
        foreign = self.root / "foreign-download-winner"
        foreign_bytes = b"foreign winner after verified link"
        foreign.write_bytes(foreign_bytes)
        real_link = download_module.os.link
        replaced = False

        def replace_after_publish(source: object, target: object) -> None:
            nonlocal replaced
            real_link(source, target)
            if Path(target) == self.destination:
                replaced = True
                foreign.replace(self.destination)

        with _RangeServer() as server:
            with patch.object(download_module.os, "link", side_effect=replace_after_publish):
                with self.assertRaisesRegex(DownloadError, "identity|preserved|publish"):
                    download_file(
                        server.url,
                        self.destination,
                        expected_sha256=PAYLOAD_SHA256,
                    )

        self.assertTrue(replaced)
        self.assertEqual(self.destination.read_bytes(), foreign_bytes)
        self.assertEqual(self.partial.read_bytes(), PAYLOAD)
        self.assertEqual(list(self.root.glob(".*.verified*")), [])

    def test_foreign_download_pin_replacement_after_link_is_preserved(self) -> None:
        foreign = self.root / "foreign-download-pin"
        foreign_bytes = b"foreign private pin"
        foreign.write_bytes(foreign_bytes)
        real_link = download_module.os.link
        foreign_pin: Path | None = None

        def replace_pin_after_link(source: object, target: object) -> None:
            nonlocal foreign_pin
            real_link(source, target)
            target_path = Path(target)
            if target_path.name.endswith(".verified.pin") and foreign_pin is None:
                foreign_pin = target_path
                foreign.replace(target_path)

        with _RangeServer() as server:
            with patch.object(download_module.os, "link", side_effect=replace_pin_after_link):
                with self.assertRaisesRegex(DownloadError, "identity|pin|changed"):
                    download_file(server.url, self.destination)

        self.assertIsNotNone(foreign_pin)
        assert foreign_pin is not None
        self.assertEqual(foreign_pin.read_bytes(), foreign_bytes)
        self.assertEqual(self.partial.read_bytes(), PAYLOAD)
        self.assertFalse(self.destination.exists())

    def test_valid_416_complete_partial_publishes_and_incomplete_retries_once(self) -> None:
        self.partial.write_bytes(PAYLOAD)
        with _RangeServer(mode="range-416") as server:
            complete = download_file(server.url, self.destination)
        self.assertEqual(complete.sha256, PAYLOAD_SHA256)
        self.assertEqual(len(server.requests), 1)

        self.destination.unlink()
        self.partial.write_bytes(PAYLOAD[:89])
        with _RangeServer(mode="range-416") as server:
            restarted = download_file(server.url, self.destination)
        self.assertEqual(len(server.requests), 2)
        self.assertEqual(server.requests[0]["Range"], "bytes=89-")
        self.assertNotIn("Range", server.requests[1])
        self.assertTrue(restarted.restarted)
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_accepted_416_http_error_is_closed_exactly_once(self) -> None:
        self.partial.write_bytes(PAYLOAD)
        headers = Message()
        headers["Content-Range"] = f"bytes */{len(PAYLOAD)}"
        error = _CloseTrackingHTTPError(
            "https://example.test/archive?access_token=error-secret#fragment-secret",
            416,
            headers,
        )

        with patch.object(download_module, "urlopen", side_effect=error):
            result = download_file("https://example.test/archive", self.destination)

        self.assertEqual(error.close_count, 1)
        self.assertEqual(result.sha256, PAYLOAD_SHA256)

    def test_retryable_416_http_error_is_closed_exactly_once(self) -> None:
        self.partial.write_bytes(PAYLOAD[:83])
        headers = Message()
        headers["Content-Range"] = f"bytes */{len(PAYLOAD)}"
        error = _CloseTrackingHTTPError(
            "https://example.test/archive?sig=retry-secret#retry-fragment",
            416,
            headers,
        )
        real_urlopen = download_module.urlopen
        calls = 0

        def fail_once_then_open(*args: object, **kwargs: object):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise error
            return real_urlopen(*args, **kwargs)

        with _RangeServer(mode="range-416") as server:
            with patch.object(download_module, "urlopen", side_effect=fail_once_then_open):
                result = download_file(server.url, self.destination)

        self.assertEqual(error.close_count, 1)
        self.assertTrue(result.restarted)
        self.assertEqual(self.destination.read_bytes(), PAYLOAD)

    def test_terminal_http_error_is_closed_once_and_diagnostics_hide_secrets(self) -> None:
        error = _CloseTrackingHTTPError(
            "https://example.test/archive?X-Amz-Signature=terminal-secret"
            "#terminal-fragment",
            503,
        )

        with patch.object(download_module, "urlopen", side_effect=error):
            with self.assertRaises(DownloadError) as caught:
                download_file(
                    "https://user:password@example.test/archive"
                    "?access_token=source-secret#source-fragment",
                    self.destination,
                )

        self.assertEqual(error.close_count, 1)
        rendered = str(caught.exception)
        for secret in (
            "user",
            "password",
            "terminal-secret",
            "terminal-fragment",
            "source-secret",
            "source-fragment",
        ):
            self.assertNotIn(secret, rendered)

    def test_base_exceptions_close_partial_descriptor_and_are_not_converted(self) -> None:
        prefix = PAYLOAD[:61]
        real_fdopen = download_module.os.fdopen
        signal_cases = (KeyboardInterrupt("interrupt"), SystemExit("exit"))

        for signal in signal_cases:
            with self.subTest(signal=type(signal).__name__):
                self.partial.unlink(missing_ok=True)
                self.destination.unlink(missing_ok=True)
                streams: list[_CloseTrackingStream] = []
                response = _InterruptingResponse(
                    "https://example.test/archive?sig=response-secret#response-fragment",
                    signal,
                    prefix,
                )

                def tracking_fdopen(*args: object, **kwargs: object):
                    tracked = _CloseTrackingStream(real_fdopen(*args, **kwargs))
                    streams.append(tracked)
                    return tracked

                with (
                    patch.object(download_module, "urlopen", return_value=response),
                    patch.object(download_module.os, "fdopen", side_effect=tracking_fdopen),
                ):
                    with self.assertRaises(type(signal)) as caught:
                        download_file("https://example.test/archive", self.destination)

                observed_close_count = streams[0].close_count
                if observed_close_count == 0:
                    streams[0].close()
                self.assertIs(caught.exception, signal)
                self.assertEqual(observed_close_count, 1)
                self.assertEqual(response.close_count, 1)
                self.assertEqual(self.partial.read_bytes(), prefix)
                self.assertFalse(self.destination.exists())

    def test_destination_and_partial_symlinks_are_rejected(self) -> None:
        outside = self.root / "outside"
        outside.write_bytes(b"outside")
        for path in (self.destination, self.partial):
            with self.subTest(path=path.name):
                try:
                    path.symlink_to(outside)
                except (OSError, NotImplementedError) as error:
                    self.skipTest(f"symlinks unavailable: {error}")
                with self.assertRaisesRegex(DownloadError, "symlink|regular"):
                    download_file("http://127.0.0.1:1/not-requested", self.destination)
                path.unlink()

    def test_url_sanitization_removes_credentials_from_values_and_errors(self) -> None:
        sanitized = sanitize_url(
            "https://user:password@example.test/archive"
            "?access_token=query-secret#oauth-fragment-secret"
        )
        self.assertEqual(sanitized, "https://example.test/archive")
        with self.assertRaises(DownloadError) as caught:
            download_file(
                "http://user:password@127.0.0.1:1/archive",
                self.destination,
                timeout=0.25,
            )
        rendered = str(caught.exception)
        self.assertNotIn("user", rendered)
        self.assertNotIn("password", rendered)

    def test_result_urls_drop_signed_queries_and_redirect_fragments(self) -> None:
        source_secrets = (
            "source-signature-secret",
            "source-oauth-secret",
            "source-fragment-secret",
        )
        with _RangeServer(mode="redirect-secret") as server:
            source_url = (
                f"{server.url}?X-Amz-Signature={source_secrets[0]}"
                f"&access_token={source_secrets[1]}#{source_secrets[2]}"
            )
            result = download_file(source_url, self.destination)

        self.assertEqual(result.source_url, server.url)
        self.assertEqual(
            result.final_response_url,
            server.url.replace("/archive.zip", "/redirected.zip"),
        )
        serialized = json.dumps(result.to_dict())
        for secret in (*source_secrets, "redirect-query-secret", "redirect-fragment-secret"):
            self.assertNotIn(secret, serialized)

    def test_redirected_http_error_diagnostics_drop_query_and_fragment_secrets(self) -> None:
        with _RangeServer(mode="redirect-error") as server:
            with self.assertRaises(DownloadError) as caught:
                download_file(
                    f"{server.url}?access_token=source-error-secret"
                    "#source-error-fragment",
                    self.destination,
                )

        rendered = str(caught.exception)
        for secret in (
            "source-error-secret",
            "source-error-fragment",
            "redirect-query-secret",
            "redirect-fragment-secret",
        ):
            self.assertNotIn(secret, rendered)


@contextmanager
def _null_context():
    yield


class SourceManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "source"
        self.root.mkdir()
        (self.root / "z.csv").write_bytes(b"z-data")
        (self.root / "nested").mkdir()
        (self.root / "nested" / "b.csv").write_bytes(b"b-data")
        (self.root / "nested" / "a.csv").write_bytes(b"a-data")
        self.registry = load_registry()

    def official_manifest(self):
        spec = self.registry["nbaiot"]
        return build_source_manifest(
            self.root,
            spec,
            acquisition_method="official-download",
            acquisition_url=spec.download_url,
        )

    def test_recursive_files_have_stable_posix_order_sizes_and_hashes(self) -> None:
        manifest = self.official_manifest()

        self.assertEqual(
            [record.relative_path for record in manifest.files],
            ["nested/a.csv", "nested/b.csv", "z.csv"],
        )
        self.assertEqual(
            [record.sha256 for record in manifest.files],
            [
                hashlib.sha256(b"a-data").hexdigest(),
                hashlib.sha256(b"b-data").hexdigest(),
                hashlib.sha256(b"z-data").hexdigest(),
            ],
        )
        self.assertEqual(manifest.total_bytes, len(b"a-datab-dataz-data"))
        self.assertEqual(
            [record.mtime_ns for record in manifest.files],
            [
                (self.root / "nested" / "a.csv").stat().st_mtime_ns,
                (self.root / "nested" / "b.csv").stat().st_mtime_ns,
                (self.root / "z.csv").stat().st_mtime_ns,
            ],
        )

    def test_source_changed_during_hash_is_rejected(self) -> None:
        target = self.root / "nested" / "a.csv"
        real_hash_file = manifest_module._hash_file

        def mutate_after_hash(path: Path) -> tuple[str, tuple[int, ...]]:
            digest = real_hash_file(path)
            if path == target:
                path.write_bytes(b"changed after hashing")
            return digest

        with patch.object(manifest_module, "_hash_file", side_effect=mutate_after_hash):
            with self.assertRaisesRegex(SourceManifestError, "changed.*hash"):
                self.official_manifest()

    def test_late_added_file_invalidates_complete_tree_snapshot(self) -> None:
        real_content_digest = manifest_module._manifest_content_digest
        mutated = False

        def add_file_before_final_validation(value: dict[str, object]) -> str:
            nonlocal mutated
            if not mutated:
                mutated = True
                (self.root / "nested" / "late.csv").write_bytes(b"late")
            return real_content_digest(value)

        with patch.object(
            manifest_module,
            "_manifest_content_digest",
            side_effect=add_file_before_final_validation,
        ):
            with self.assertRaisesRegex(SourceManifestError, "changed|snapshot|tree|directory"):
                self.official_manifest()

    def test_late_removed_file_invalidates_complete_tree_snapshot(self) -> None:
        real_content_digest = manifest_module._manifest_content_digest
        mutated = False

        def remove_file_before_final_validation(value: dict[str, object]) -> str:
            nonlocal mutated
            if not mutated:
                mutated = True
                (self.root / "nested" / "a.csv").unlink()
            return real_content_digest(value)

        with patch.object(
            manifest_module,
            "_manifest_content_digest",
            side_effect=remove_file_before_final_validation,
        ):
            with self.assertRaisesRegex(SourceManifestError, "changed|snapshot|tree|file"):
                self.official_manifest()

    def test_late_replaced_file_invalidates_complete_tree_snapshot(self) -> None:
        real_content_digest = manifest_module._manifest_content_digest
        target = self.root / "nested" / "a.csv"
        replacement = self.root / "replacement.csv"
        replacement.write_bytes(b"a-data")
        mutated = False

        def replace_file_before_final_validation(value: dict[str, object]) -> str:
            nonlocal mutated
            if not mutated:
                mutated = True
                os.replace(replacement, target)
            return real_content_digest(value)

        with patch.object(
            manifest_module,
            "_manifest_content_digest",
            side_effect=replace_file_before_final_validation,
        ):
            with self.assertRaisesRegex(SourceManifestError, "changed|snapshot|tree|identity"):
                self.official_manifest()

    def test_file_and_directory_symlinks_are_rejected(self) -> None:
        cases = (("file-link", self.root / "z.csv"), ("dir-link", self.root / "nested"))
        for name, target in cases:
            link = self.root / name
            try:
                link.symlink_to(target, target_is_directory=target.is_dir())
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"symlinks unavailable: {error}")
            with self.subTest(name=name):
                with self.assertRaisesRegex(SourceManifestError, "symlink"):
                    self.official_manifest()
            link.unlink()

    def test_ancestor_directory_swap_before_hashing_is_rejected(self) -> None:
        external = self.root.parent / "external-source"
        external.mkdir()
        (external / "a.csv").write_bytes(b"external-a")
        (external / "b.csv").write_bytes(b"external-b")
        probe = self.root.parent / "directory-symlink-probe"
        try:
            probe.symlink_to(external, target_is_directory=True)
            probe.unlink()
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")

        real_enumerate = manifest_module._enumerate_files
        original_nested = self.root / "nested"
        displaced_nested = self.root / "nested-before-swap"

        def swap_ancestor_after_enumeration(root: Path):
            records = real_enumerate(root)
            original_nested.replace(displaced_nested)
            original_nested.symlink_to(external, target_is_directory=True)
            return records

        with patch.object(
            manifest_module,
            "_enumerate_files",
            side_effect=swap_ancestor_after_enumeration,
        ):
            with self.assertRaisesRegex(
                SourceManifestError,
                "ancestor|directory|changed|symlink",
            ):
                self.official_manifest()

    def test_official_nbaiot_metadata_and_url_are_validated(self) -> None:
        spec = self.registry["nbaiot"]
        manifest = self.official_manifest()
        self.assertEqual(manifest.dataset_name, "nbaiot")
        self.assertEqual(manifest.project_url, spec.project_url)
        self.assertEqual(manifest.doi, spec.doi)
        self.assertEqual(manifest.license_name, spec.license_name)
        self.assertEqual(manifest.acquisition_method, "official-download")
        self.assertEqual(manifest.acquisition_url, spec.download_url)

        for url in (
            "https://user:password@archive.ics.uci.edu/static/public/442/file.zip",
            "http://archive.ics.uci.edu/static/public/442/file.zip",
            "https://example.test/not-official.zip",
        ):
            with self.subTest(url=url):
                with self.assertRaisesRegex(SourceManifestError, "credential|HTTPS|official"):
                    build_source_manifest(
                        self.root,
                        spec,
                        acquisition_method="official-download",
                        acquisition_url=url,
                    )

    def test_botiot_is_manual_only_and_never_records_acquisition_url(self) -> None:
        spec = self.registry["botiot"]
        manifest = build_source_manifest(
            self.root,
            spec,
            acquisition_method="manual-local-source",
        )
        self.assertEqual(manifest.dataset_name, "botiot")
        self.assertEqual(manifest.project_url, spec.project_url)
        self.assertEqual(manifest.acquisition_method, "manual-local-source")
        self.assertIsNone(manifest.acquisition_url)

        for method, url in (
            ("official-download", None),
            ("manual-local-source", "https://sharepoint.example/login"),
        ):
            with self.subTest(method=method):
                with self.assertRaisesRegex(SourceManifestError, "BoT-IoT|manual|URL"):
                    build_source_manifest(
                        self.root,
                        spec,
                        acquisition_method=method,
                        acquisition_url=url,
                    )

    def test_json_is_deterministic_and_round_trips(self) -> None:
        first = self.official_manifest()
        second = self.official_manifest()
        first_bytes = manifest_json_bytes(first)
        second_bytes = manifest_json_bytes(second)

        self.assertEqual(first, second)
        self.assertEqual(first_bytes, second_bytes)
        self.assertTrue(first_bytes.endswith(b"\n"))
        self.assertEqual(
            manifest_module.SourceManifest.from_dict(json.loads(first_bytes)),
            first,
        )
        self.assertEqual(json.loads(json.dumps(first.to_dict())), first.to_dict())

    def test_mtime_is_serialized_validated_and_changes_content_digest(self) -> None:
        first = self.official_manifest()
        target = self.root / "z.csv"
        original = target.stat().st_mtime_ns
        requested = original + 2_000_000_000
        os.utime(target, ns=(requested, requested))
        second = self.official_manifest()

        first_record = next(record for record in first.files if record.relative_path == "z.csv")
        second_record = next(record for record in second.files if record.relative_path == "z.csv")
        self.assertEqual(getattr(first_record, "mtime_ns", None), original)
        self.assertEqual(
            getattr(second_record, "mtime_ns", None),
            target.stat().st_mtime_ns,
        )
        self.assertNotEqual(
            getattr(first_record, "mtime_ns", None),
            getattr(second_record, "mtime_ns", None),
        )
        self.assertNotEqual(first.content_sha256, second.content_sha256)
        self.assertIn("mtime_ns", second_record.to_dict())

    def test_legacy_invalid_mtime_and_boolean_versions_are_rejected(self) -> None:
        manifest = self.official_manifest()
        serialized = manifest.to_dict()
        self.assertEqual(serialized["version"], 2)

        for invalid_version in (True, False, 1, 2.0, "2"):
            with self.subTest(version=invalid_version):
                invalid = dict(serialized)
                invalid["version"] = invalid_version
                with self.assertRaisesRegex(SourceManifestError, "version"):
                    manifest_module.SourceManifest.from_dict(invalid)

        legacy = json.loads(json.dumps(serialized))
        legacy["files"][0].pop("mtime_ns")
        with self.assertRaisesRegex(SourceManifestError, "file record|fields"):
            manifest_module.SourceManifest.from_dict(legacy)

        for invalid_mtime in (True, -1, 1.5, "1"):
            with self.subTest(mtime_ns=invalid_mtime):
                invalid = json.loads(json.dumps(serialized))
                invalid["files"][0]["mtime_ns"] = invalid_mtime
                with self.assertRaisesRegex(SourceManifestError, "mtime"):
                    manifest_module.SourceManifest.from_dict(invalid)

    def test_serialization_rejects_a_directly_constructed_credential_url(self) -> None:
        unsafe = replace(
            self.official_manifest(),
            acquisition_url="https://user:password@archive.ics.uci.edu/archive.zip",
        )

        with self.assertRaisesRegex(SourceManifestError, "credential|official"):
            manifest_json_bytes(unsafe)

    def test_manifest_write_is_immutable_and_identical_content_is_reused(self) -> None:
        manifest = self.official_manifest()
        path = self.root.parent / "source-manifest.json"
        self.assertTrue(write_source_manifest(path, manifest))
        original = path.read_bytes()
        self.assertFalse(write_source_manifest(path, manifest))
        self.assertEqual(path.read_bytes(), original)

        (self.root / "z.csv").write_bytes(b"different")
        different = self.official_manifest()
        with self.assertRaisesRegex(SourceManifestError, "different|overwrite"):
            write_source_manifest(path, different)
        self.assertEqual(path.read_bytes(), original)

    def test_manifest_concurrent_target_is_never_overwritten(self) -> None:
        manifest = self.official_manifest()
        path = self.root.parent / "source-manifest.json"
        winner = b'{"winner":true}\n'
        real_link = manifest_module.os.link

        def concurrent_publish(source: object, target: object) -> None:
            if Path(target) == path:
                Path(target).write_bytes(winner)
                raise FileExistsError(str(target))
            real_link(source, target)

        with patch.object(manifest_module.os, "link", side_effect=concurrent_publish):
            with self.assertRaisesRegex(SourceManifestError, "different|overwrite"):
                write_source_manifest(path, manifest)

        self.assertEqual(path.read_bytes(), winner)
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_manifest_temp_substitution_cannot_publish_unverified_bytes(self) -> None:
        manifest = self.official_manifest()
        path = self.root.parent / "source-manifest.json"
        displaced = self.root.parent / "owned-manifest-temp"
        attacker = b'{"attacker":true}\n'
        real_link = manifest_module.os.link
        substituted = False

        def substitute_temporary(source: object, target: object) -> None:
            nonlocal substituted
            source_path = Path(source)
            if not substituted and source_path.name.endswith(".tmp"):
                substituted = True
                source_path.replace(displaced)
                source_path.write_bytes(attacker)
            real_link(source, target)

        with patch.object(manifest_module.os, "link", side_effect=substitute_temporary):
            with self.assertRaisesRegex(SourceManifestError, "changed|identity|temporary|pin"):
                write_source_manifest(path, manifest)

        self.assertTrue(substituted)
        self.assertFalse(path.exists())
        if displaced.exists():
            self.assertEqual(displaced.read_bytes(), manifest_json_bytes(manifest))
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.pin")), [])

    def test_manifest_private_pin_substitution_preserves_unverified_final(self) -> None:
        manifest = self.official_manifest()
        path = self.root.parent / "source-manifest.json"
        displaced = self.root.parent / "owned-manifest-pin"
        attacker = b'{"attacker":true}\n'
        real_link = manifest_module.os.link
        substituted = False

        def substitute_pin(source: object, target: object) -> None:
            nonlocal substituted
            source_path = Path(source)
            if (
                not substituted
                and source_path.name.endswith(".pin")
                and Path(target) == path
            ):
                substituted = True
                source_path.replace(displaced)
                source_path.write_bytes(attacker)
            real_link(source, target)

        with patch.object(manifest_module.os, "link", side_effect=substitute_pin):
            with self.assertRaisesRegex(SourceManifestError, "identity|publish|verified|pin"):
                write_source_manifest(path, manifest)

        self.assertTrue(substituted)
        self.assertEqual(path.read_bytes(), attacker)
        self.assertEqual(displaced.read_bytes(), manifest_json_bytes(manifest))

    def test_foreign_manifest_replacement_after_link_is_preserved(self) -> None:
        manifest = self.official_manifest()
        path = self.root.parent / "source-manifest.json"
        foreign = self.root.parent / "foreign-manifest-winner"
        foreign_bytes = b'{"foreign":true}\n'
        foreign.write_bytes(foreign_bytes)
        real_link = manifest_module.os.link
        replaced = False

        def replace_after_publish(source: object, target: object) -> None:
            nonlocal replaced
            real_link(source, target)
            if Path(target) == path:
                replaced = True
                foreign.replace(path)

        with patch.object(manifest_module.os, "link", side_effect=replace_after_publish):
            with self.assertRaisesRegex(SourceManifestError, "identity|preserved|pin"):
                write_source_manifest(path, manifest)

        self.assertTrue(replaced)
        self.assertEqual(path.read_bytes(), foreign_bytes)
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.pin")), [])

    def test_foreign_manifest_pin_replacement_after_link_is_preserved(self) -> None:
        manifest = self.official_manifest()
        path = self.root.parent / "source-manifest.json"
        foreign = self.root.parent / "foreign-manifest-pin"
        foreign_bytes = b'{"foreign-pin":true}\n'
        foreign.write_bytes(foreign_bytes)
        real_link = manifest_module.os.link
        foreign_pin: Path | None = None

        def replace_pin_after_link(source: object, target: object) -> None:
            nonlocal foreign_pin
            real_link(source, target)
            target_path = Path(target)
            if target_path.name.endswith(".pin") and foreign_pin is None:
                foreign_pin = target_path
                foreign.replace(target_path)

        with patch.object(manifest_module.os, "link", side_effect=replace_pin_after_link):
            with self.assertRaisesRegex(SourceManifestError, "identity|pin|changed"):
                write_source_manifest(path, manifest)

        self.assertIsNotNone(foreign_pin)
        assert foreign_pin is not None
        self.assertEqual(foreign_pin.read_bytes(), foreign_bytes)
        self.assertFalse(path.exists())

    def test_existing_manifest_replacement_during_pinned_read_is_rejected(self) -> None:
        manifest = self.official_manifest()
        path = self.root.parent / "source-manifest.json"
        displaced = self.root.parent / "original-source-manifest.json"
        payload = manifest_json_bytes(manifest)
        self.assertTrue(write_source_manifest(path, manifest))
        real_open = manifest_module.os.open
        substituted = False

        def substitute_after_open(
            opened_path: object,
            flags: int,
            mode: int = 0o777,
        ) -> int:
            nonlocal substituted
            descriptor = real_open(opened_path, flags, mode)
            if Path(opened_path) == path and not substituted:
                substituted = True
                try:
                    path.replace(displaced)
                    path.write_bytes(payload)
                except OSError:
                    os.close(descriptor)
                    raise
            return descriptor

        with patch.object(manifest_module.os, "open", side_effect=substitute_after_open):
            with self.assertRaisesRegex(
                SourceManifestError,
                "changed|identity|replacement|safely open",
            ):
                write_source_manifest(path, manifest)

        self.assertTrue(substituted)
        if displaced.exists():
            self.assertEqual(displaced.read_bytes(), payload)
        self.assertEqual(path.read_bytes(), payload)

    def test_manifest_base_exceptions_close_and_clean_private_files(self) -> None:
        manifest = self.official_manifest()
        real_create = manifest_module._create_temporary
        real_fsync = manifest_module.os.fsync

        for phase, signal in (
            ("write", KeyboardInterrupt("manifest-write-interrupt")),
            ("fsync", SystemExit("manifest-fsync-exit")),
        ):
            with self.subTest(phase=phase):
                path = self.root.parent / f"{phase}-source-manifest.json"
                descriptors: list[int] = []

                def track_create(target: Path) -> tuple[Path, int]:
                    temporary, descriptor = real_create(target)
                    descriptors.append(descriptor)
                    return temporary, descriptor

                def interrupt_fsync(descriptor: int) -> None:
                    if phase == "fsync" and descriptor in descriptors:
                        raise signal
                    real_fsync(descriptor)

                write_effect = signal if phase == "write" else None
                with (
                    patch.object(
                        manifest_module,
                        "_create_temporary",
                        side_effect=track_create,
                    ),
                    patch.object(
                        manifest_module,
                        "_write_all",
                        side_effect=write_effect,
                        wraps=None if write_effect is not None else manifest_module._write_all,
                    ),
                    patch.object(
                        manifest_module.os,
                        "fsync",
                        side_effect=interrupt_fsync,
                    ),
                ):
                    with self.assertRaises(type(signal)) as caught:
                        write_source_manifest(path, manifest)

                self.assertIs(caught.exception, signal)
                self.assertEqual(len(descriptors), 1)
                try:
                    os.fstat(descriptors[0])
                except OSError:
                    descriptor_closed = True
                else:
                    descriptor_closed = False
                    os.close(descriptors[0])
                self.assertTrue(descriptor_closed)
                self.assertFalse(path.exists())
                self.assertEqual(
                    list(path.parent.glob(f".{path.name}.*.tmp")),
                    [],
                )
                self.assertEqual(
                    list(path.parent.glob(f".{path.name}.*.pin")),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
