"""Reject mounts whose filesystem type cannot support a cross-host atomic create.

Filesystem type is the coherency model: a local type can only ever be mounted by
one host, and a network type declares its own semantics. Checking the type is
cheap, runs before any I/O, and produces a far better error message than a
failed write.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from .contracts import StorageCapabilityError

logger = logging.getLogger(__name__)

_MOUNTINFO_PATH = "/proc/self/mountinfo"

# Declare cross-host cache coherency.
_COHERENT = frozenset(
    {
        "nfs",
        "nfs3",
        "nfs4",
        "cifs",
        "smb3",
        "glusterfs",
        "cephfs",
        "ceph",
        "lustre",
        "gfs2",
        "ocfs2",
        "beegfs",
    }
)

# Only one host can mount these, so "shared between containers on one host" is
# the only possible topology and link() is kernel-atomic there.
_LOCAL = frozenset(
    {
        "ext2",
        "ext3",
        "ext4",
        "xfs",
        "btrfs",
        "zfs",
        "overlay",
        "overlayfs",
        "tmpfs",
        "f2fs",
        "jfs",
        "reiserfs",
        "bcachefs",
        "ntfs",
        "ntfs3",
        "vfat",
        "exfat",
        "erofs",
        "squashfs",
    }
)

# Object-storage gateways: no link(), and their O_EXCL is not atomic between
# clients because the underlying store has no such primitive.
_BROKEN = frozenset(
    {
        "fuse.s3fs",
        "fuse.gcsfuse",
        "fuse.rclone",
        "fuse.blobfuse",
        "fuse.blobfuse2",
        "fuse.juicefs",
        "fuse.goofys",
        "s3fs",
        "gcsfuse",
        "rclone",
        "blobfuse",
        "juicefs",
    }
)

_NFS_VERSION = re.compile(r"\b(?:nfs)?vers=(\d+)")
_CIFS_VERSION = re.compile(r"\bvers=(\d+)(?:\.(\d+))?")

_ESCAPES = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}


@dataclass(frozen=True)
class MountEntry:
    mount_point: str
    fs_type: str
    options: str


def _unescape(field: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(field):
        if field[i] == "\\" and field[i + 1 : i + 4] in _ESCAPES:
            out.append(_ESCAPES[field[i + 1 : i + 4]])
            i += 4
        else:
            out.append(field[i])
            i += 1
    return "".join(out)


def parse_mountinfo(text: str) -> list[MountEntry]:
    entries: list[MountEntry] = []
    for line in text.splitlines():
        if " - " not in line:
            continue
        left, _, right = line.partition(" - ")
        left_fields = left.split(" ")
        right_fields = right.split(" ")
        if len(left_fields) < 6 or len(right_fields) < 3:
            continue
        options = ",".join(f for f in (left_fields[5], right_fields[2]) if f)
        entries.append(
            MountEntry(
                mount_point=_unescape(left_fields[4]),
                fs_type=right_fields[0],
                options=options,
            )
        )
    return entries


def find_mount_for(path: str, entries: list[MountEntry]) -> MountEntry | None:
    """Return the entry whose mount point is the longest prefix of `path`."""
    best: MountEntry | None = None
    for entry in entries:
        mp = entry.mount_point.rstrip("/") or "/"
        if path == mp or path.startswith(mp if mp.endswith("/") else mp + "/"):
            if best is None or len(mp) > len(best.mount_point.rstrip("/") or "/"):
                best = entry
    return best


def classify(entry: MountEntry) -> tuple[str, str]:
    """Return (verdict, reason) where verdict is 'ok', 'refuse', or 'warn'."""
    fs_type = entry.fs_type.lower()

    if fs_type in _BROKEN:
        return (
            "refuse",
            f"filesystem type '{entry.fs_type}' is an object-storage gateway: it "
            "does not implement hard links and its O_EXCL is not atomic between "
            "clients",
        )

    if fs_type in ("nfs", "nfs3", "nfs4"):
        match = _NFS_VERSION.search(entry.options)
        if match and int(match.group(1)) < 3:
            return (
                "refuse",
                f"NFS protocol version {match.group(1)} does not provide the "
                "close-to-open coherency an atomic create-if-absent relies on; "
                "mount with vers=3 or later",
            )
        return "ok", ""

    if fs_type in ("cifs", "smb3", "smb2"):
        match = _CIFS_VERSION.search(entry.options)
        if match and int(match.group(1)) < 3:
            return (
                "warn",
                f"SMB protocol version {match.group(0)} is older than 3.0; "
                "atomic create-if-absent is only verified on SMB3",
            )
        return "ok", ""

    if fs_type in _COHERENT or fs_type in _LOCAL:
        return "ok", ""

    return (
        "warn",
        f"unrecognised filesystem type '{entry.fs_type}'; whether it supports a "
        "cross-host atomic create could not be determined from the mount table",
    )


def check_mount(root: str, *, allow_unsafe: bool) -> None:
    """Raise StorageCapabilityError when `root` sits on a known-broken mount."""
    try:
        with open(_MOUNTINFO_PATH, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        logger.info(
            "Storage mount check skipped: %s is unavailable on this platform; "
            "relying on the startup atomic-create probe instead",
            _MOUNTINFO_PATH,
        )
        return

    resolved = os.path.realpath(root)
    entry = find_mount_for(resolved, parse_mountinfo(text))
    if entry is None:
        logger.warning(
            "Storage mount check: no mount table entry covers %s; relying on the "
            "startup atomic-create probe instead",
            resolved,
        )
        return

    verdict, reason = classify(entry)
    if verdict == "ok":
        logger.info(
            "Storage mount check: %s is on %s (%s) — suitable for the job lease",
            resolved,
            entry.mount_point,
            entry.fs_type,
        )
        return

    message = (
        f"STORAGE__FS__ROOT={root} is on a mount that cannot safely hold the "
        f"claims/<job_id> processing lease: {reason}. Without an atomic "
        "create-if-absent two workers can process the same job concurrently."
    )
    if verdict == "warn":
        logger.warning("%s Continuing — the atomic-create probe runs next.", message)
        return

    if allow_unsafe:
        logger.warning(
            "%s Continuing because STORAGE__FS__ALLOW_UNSAFE_MOUNT is set.", message
        )
        return
    raise StorageCapabilityError(
        message + " Use a mount whose filesystem implements hard links (NFSv3+/NFSv4, "
        "SMB3, a cluster filesystem, or any local filesystem shared between "
        "containers on one host), or set STORAGE__FS__ALLOW_UNSAFE_MOUNT=true "
        "to override this refusal."
    )
