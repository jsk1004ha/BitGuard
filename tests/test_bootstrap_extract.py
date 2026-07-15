from __future__ import annotations

import io
import os
import stat
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

from bitguard_bnn.bootstrap.extract import (
    ArchiveExtractionError,
    MissingArchiveToolError,
    extract_rar,
    extract_zip,
    parse_7z_listing,
)


def _write_zip(path: Path, entries: list[tuple[zipfile.ZipInfo | str, bytes]]) -> None:
    encrypted = any(
        isinstance(name, zipfile.ZipInfo) and bool(name.flag_bits & 0x1)
        for name, _ in entries
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in entries:
                archive.writestr(name, payload)
    if encrypted:
        payload = bytearray(path.read_bytes())
        offset = 0
        while True:
            offset = payload.find(b"PK\x03\x04", offset)
            if offset < 0:
                break
            payload[offset + 6] |= 0x1
            offset += 4
        offset = 0
        while True:
            offset = payload.find(b"PK\x01\x02", offset)
            if offset < 0:
                break
            payload[offset + 8] |= 0x1
            offset += 4
        path.write_bytes(payload)


class SafeZipExtractionTest(unittest.TestCase):
    def _assert_rejected_without_destination(
        self, entries: list[tuple[zipfile.ZipInfo | str, bytes]], pattern: str
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            destination = root / "output"
            _write_zip(archive, entries)
            with self.assertRaisesRegex(ArchiveExtractionError, pattern):
                extract_zip(archive, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(sorted(item.name for item in root.iterdir()), ["source.zip"])

    def test_rejects_parent_traversal_before_writing(self) -> None:
        self._assert_rejected_without_destination(
            [("safe.csv", b"x\n1\n"), ("../escape.csv", b"bad")], "unsafe archive path"
        )

    def test_rejects_absolute_windows_unc_device_and_posix_paths(self) -> None:
        names = (
            "/absolute.csv",
            "C:/absolute.csv",
            r"C:\absolute.csv",
            r"\\server\share\absolute.csv",
            r"\\?\C:\absolute.csv",
            r"\\.\PhysicalDrive0",
        )
        for name in names:
            with self.subTest(name=name):
                self._assert_rejected_without_destination([(name, b"bad")], "unsafe archive path")

    def test_rejects_backslash_traversal(self) -> None:
        self._assert_rejected_without_destination(
            [(r"folder\..\escape.csv", b"bad")], "unsafe archive path"
        )

    def test_rejects_duplicate_normalized_and_casefolded_names(self) -> None:
        for entries in (
            [("a.csv", b"1"), ("a.csv", b"2")],
            [("A.csv", b"1"), ("a.csv", b"2")],
            [
                ("caf\N{LATIN SMALL LETTER E WITH ACUTE}.csv", b"1"),
                ("cafe\N{COMBINING ACUTE ACCENT}.csv", b"2"),
            ],
        ):
            with self.subTest(entries=[str(item[0]) for item in entries]):
                self._assert_rejected_without_destination(entries, "duplicate archive destination")

    def test_rejects_links_special_modes_encryption_and_ambiguous_names(self) -> None:
        symlink = zipfile.ZipInfo("link.csv")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        fifo = zipfile.ZipInfo("pipe.csv")
        fifo.create_system = 3
        fifo.external_attr = (stat.S_IFIFO | 0o600) << 16
        encrypted = zipfile.ZipInfo("encrypted.csv")
        encrypted.flag_bits |= 0x1
        for entry, pattern in (
            (symlink, "link or special"),
            (fifo, "link or special"),
            (encrypted, "encrypted"),
            ("folder//file.csv", "unsafe archive path"),
            ("folder/./file.csv", "unsafe archive path"),
            ("CON.csv", "unsafe archive path"),
            ("file.csv:stream", "unsafe archive path"),
            ("file.csv. ", "unsafe archive path"),
        ):
            with self.subTest(entry=str(entry)):
                self._assert_rejected_without_destination([(entry, b"bad")], pattern)

    def test_validates_every_entry_before_any_write(self) -> None:
        self._assert_rejected_without_destination(
            [("first/large.csv", b"good" * 100), ("../../late.csv", b"bad")],
            "unsafe archive path",
        )

    def test_rejects_file_directory_prefix_conflicts_before_writing(self) -> None:
        self._assert_rejected_without_destination(
            [("prefix", b"file"), ("PREFIX/child.csv", b"row")],
            "conflicts with a descendant",
        )

    def test_declared_size_preflight_fails_before_destination_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            destination = root / "output"
            _write_zip(archive, [("data.csv", b"123456")])
            with self.assertRaisesRegex(ArchiveExtractionError, "available=5"):
                extract_zip(archive, destination, disk_free_fn=lambda _: 5)
            self.assertFalse(destination.exists())

    def test_streaming_limit_rejects_metadata_lie_and_cleans_private_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            destination = root / "output"
            _write_zip(archive, [("data.csv", b"0123456789")])

            real_open = zipfile.ZipFile.open

            def lied_open(instance, member, *args, **kwargs):
                stream = real_open(instance, member, *args, **kwargs)
                if getattr(member, "filename", member) == "data.csv":
                    return io.BytesIO(stream.read() + b"extra")
                return stream

            with (
                patch("zipfile.ZipFile.open", lied_open),
                self.assertRaisesRegex(ArchiveExtractionError, "declared byte limit"),
            ):
                extract_zip(archive, destination, chunk_size=3)
            self.assertFalse(destination.exists())
            self.assertEqual(sorted(item.name for item in root.iterdir()), ["source.zip"])

    def test_private_temp_mutation_during_link_is_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            destination = root / "output"
            _write_zip(archive, [("data.csv", b"trusted")])
            real_link = os.link

            def mutating_link(source, target, *args, **kwargs):
                result = real_link(source, target, *args, **kwargs)
                source_path = Path(source)
                if source_path.name.endswith(".partial"):
                    source_path.write_bytes(b"changed")
                return result

            with (
                patch("bitguard_bnn.bootstrap.extract.os.link", mutating_link),
                self.assertRaisesRegex(ArchiveExtractionError, "content changed"),
            ):
                extract_zip(archive, destination)
            self.assertFalse(destination.exists())

    def test_staged_file_mutation_during_final_publication_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            destination = root / "output"
            _write_zip(archive, [("data.csv", b"trusted")])
            real_link = os.link

            def mutating_link(source, target, *args, **kwargs):
                result = real_link(source, target, *args, **kwargs)
                source_path = Path(source)
                if source_path.name == "data.csv":
                    source_path.write_bytes(b"changed")
                return result

            with (
                patch("bitguard_bnn.bootstrap.extract.os.link", mutating_link),
                self.assertRaisesRegex(ArchiveExtractionError, "content changed"),
            ):
                extract_zip(archive, destination)
            self.assertFalse(destination.exists())

    def test_extracts_regular_files_with_stable_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            destination = root / "output"
            _write_zip(
                archive,
                [("nested/one.csv", b"a,b\n1,2\n"), ("two.csv", b"x\n3\n")],
            )
            result = extract_zip(archive, destination, chunk_size=2)
            self.assertEqual(result.extractor, "zipfile")
            self.assertEqual(result.files, ("nested/one.csv", "two.csv"))
            self.assertEqual(result.total_bytes, 12)
            self.assertEqual((destination / "nested" / "one.csv").read_bytes(), b"a,b\n1,2\n")
            self.assertEqual(result.as_dict()["destination"], str(destination.resolve()))

    def test_publishes_deep_implicit_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            destination = root / "output"
            _write_zip(archive, [("one/two/three/data.csv", b"x\n1\n")])
            extract_zip(archive, destination)
            self.assertEqual(
                (destination / "one" / "two" / "three" / "data.csv").read_bytes(),
                b"x\n1\n",
            )

    def test_rejects_existing_or_symlinked_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            _write_zip(archive, [("data.csv", b"x")])
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ArchiveExtractionError, "must not already exist"):
                extract_zip(archive, existing)
            if hasattr(os, "symlink"):
                linked = root / "linked"
                try:
                    os.symlink(existing, linked, target_is_directory=True)
                except OSError:
                    return
                with self.assertRaisesRegex(ArchiveExtractionError, "must not already exist"):
                    extract_zip(archive, linked)


class RarExtractionTest(unittest.TestCase):
    LISTING = """
Path = source.rar
Type = Rar5
Physical Size = 42
----------
Path = nested/data.csv
Size = 7
Packed Size = 5
Folder = -
Encrypted = -
Mode = -rw-r--r--

Path = empty
Size = 0
Packed Size = 0
Folder = +
Encrypted = -
Mode = drwxr-xr-x
"""

    def test_parses_and_validates_7z_listing(self) -> None:
        entries = parse_7z_listing(self.LISTING)
        self.assertEqual([(item.path, item.size, item.is_dir) for item in entries], [
            ("nested/data.csv", 7, False),
            ("empty", 0, True),
        ])

    def test_rejects_unsafe_duplicate_encrypted_link_and_malformed_rar_entries(self) -> None:
        cases = (
            (self.LISTING.replace("nested/data.csv", "../escape.csv"), "unsafe archive path"),
            (
                self.LISTING + self.LISTING.split("----------", 1)[1],
                "duplicate archive destination",
            ),
            (self.LISTING.replace("Encrypted = -", "Encrypted = +", 1), "encrypted"),
            (self.LISTING.replace("Mode = -rw-r--r--", "Mode = lrwxrwxrwx", 1), "link or special"),
            (self.LISTING.replace("Size = 7", "Size = nope", 1), "invalid Size"),
        )
        for listing, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(
                ArchiveExtractionError, pattern
            ):
                parse_7z_listing(listing)

    def test_missing_tool_reports_exact_non_privileged_remediation(self) -> None:
        cases = (
            (
                "linux",
                "apt-get",
                ("apt-get", "install", "-y", "p7zip-full"),
            ),
            (
                "linux",
                "dnf",
                ("dnf", "install", "-y", "p7zip", "p7zip-plugins"),
            ),
            (
                "windows",
                None,
                (
                    "winget",
                    "install",
                    "--id",
                    "7zip.7zip",
                    "--exact",
                    "--source",
                    "winget",
                ),
            ),
        )
        for system, manager, expected in cases:
            with self.subTest(system=system, manager=manager):
                with tempfile.TemporaryDirectory() as directory:
                    source = Path(directory) / "source.rar"
                    source.write_bytes(b"rar")
                    with self.assertRaises(MissingArchiveToolError) as raised:
                        extract_rar(
                            source,
                            Path(directory) / "out",
                            which_fn=lambda _: None,
                            platform_name=system,
                            package_manager=manager,
                        )
                self.assertEqual(raised.exception.command, expected)
                self.assertNotIn("sudo", str(raised.exception))

    def test_system_install_requires_consent_and_rechecks_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.rar"
            source.write_bytes(b"rar")
            calls: list[list[str]] = []
            available = False

            def which(name: str) -> str | None:
                if name == "apt-get":
                    return "/usr/bin/apt-get"
                return "/usr/bin/7z" if available and name == "7z" else None

            def run(args, **kwargs):
                nonlocal available
                calls.append(list(args))
                available = True
                return subprocess.CompletedProcess(args, 0, "", "")

            with self.assertRaisesRegex(ArchiveExtractionError, "listing failed"):
                extract_rar(
                    source,
                    Path(directory) / "out",
                    install_system_tools=True,
                    which_fn=which,
                    run_fn=run,
                    platform_name="linux",
                    package_manager="apt-get",
                )
            self.assertEqual(calls[0], ["/usr/bin/apt-get", "install", "-y", "p7zip-full"])
            self.assertEqual(calls[1][:3], ["/usr/bin/7z", "l", "-slt"])
            self.assertIn("-sccUTF-8", calls[1])

    def test_uses_argument_arrays_lists_first_then_validates_result_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.rar"
            source.write_bytes(b"rar")
            destination = root / "out"
            calls: list[list[str]] = []

            def run(args, **kwargs):
                calls.append(list(args))
                if args[1] == "l":
                    return subprocess.CompletedProcess(args, 0, self.LISTING, "")
                output_arg = next(item for item in args if item.startswith("-o"))
                output = Path(output_arg[2:])
                (output / "nested").mkdir(parents=True)
                (output / "nested" / "data.csv").write_bytes(b"1234567")
                (output / "empty").mkdir()
                return subprocess.CompletedProcess(args, 0, "", "")

            result = extract_rar(
                source,
                destination,
                which_fn=lambda name: "/tools/7z" if name == "7z" else None,
                run_fn=run,
                disk_free_fn=lambda _: 100,
            )
            self.assertEqual([call[1] for call in calls], ["l", "x"])
            self.assertIn("--", calls[0])
            self.assertIn("--", calls[1])
            self.assertEqual(result.files, ("nested/data.csv",))
            self.assertEqual((destination / "nested" / "data.csv").read_bytes(), b"1234567")

    def test_rejects_unexpected_result_tree_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.rar"
            source.write_bytes(b"rar")
            destination = root / "out"

            def run(args, **kwargs):
                if args[1] == "l":
                    return subprocess.CompletedProcess(args, 0, self.LISTING, "")
                output = Path(next(item for item in args if item.startswith("-o"))[2:])
                output.mkdir(exist_ok=True)
                (output / "unexpected.csv").write_bytes(b"bad")
                return subprocess.CompletedProcess(args, 0, "", "")

            with self.assertRaisesRegex(ArchiveExtractionError, "result tree"):
                extract_rar(
                    source,
                    destination,
                    which_fn=lambda _: "/tools/7z",
                    run_fn=run,
                )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
