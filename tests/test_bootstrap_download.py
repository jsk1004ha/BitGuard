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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

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

    def test_publish_never_links_a_substituted_partial_inode(self) -> None:
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
                with self.assertRaisesRegex(DownloadError, "changed|identity"):
                    download_file(server.url, self.destination)

        self.assertTrue(substituted)
        self.assertEqual(self.partial.read_bytes(), winner)
        self.assertEqual(displaced.read_bytes(), PAYLOAD)
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
        sanitized = sanitize_url("https://user:password@example.test/archive?q=1")
        self.assertEqual(sanitized, "https://example.test/archive?q=1")
        with self.assertRaises(DownloadError) as caught:
            download_file(
                "http://user:password@127.0.0.1:1/archive",
                self.destination,
                timeout=0.25,
            )
        rendered = str(caught.exception)
        self.assertNotIn("user", rendered)
        self.assertNotIn("password", rendered)


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

    def test_source_changed_during_hash_is_rejected(self) -> None:
        target = self.root / "nested" / "a.csv"
        real_hash_file = manifest_module._hash_file

        def mutate_after_hash(path: Path) -> str:
            digest = real_hash_file(path)
            if path == target:
                path.write_bytes(b"changed after hashing")
            return digest

        with patch.object(manifest_module, "_hash_file", side_effect=mutate_after_hash):
            with self.assertRaisesRegex(SourceManifestError, "changed.*hash"):
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

        def concurrent_publish(source: object, target: object) -> None:
            Path(target).write_bytes(winner)
            raise FileExistsError(str(target))

        with patch.object(manifest_module.os, "link", side_effect=concurrent_publish):
            with self.assertRaisesRegex(SourceManifestError, "different|overwrite"):
                write_source_manifest(path, manifest)

        self.assertEqual(path.read_bytes(), winner)
        self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
