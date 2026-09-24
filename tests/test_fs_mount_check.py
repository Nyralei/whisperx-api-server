"""Mount-table classification: which filesystems can hold the job lease."""

import errno
import os

import pytest

from whisperx_api_server.config import FsStorageConfig
from whisperx_api_server.storage import mount_check
from whisperx_api_server.storage.contracts import StorageCapabilityError
from whisperx_api_server.storage.fs_store import FsObjectStore

pytestmark = pytest.mark.anyio

MOUNTINFO = """\
25 0 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw,errors=remount-ro
36 25 0:36 / /mnt rw,relatime shared:12 - ext4 /dev/sdb1 rw
40 36 0:39 / /mnt/nfs rw,relatime shared:22 - nfs4 10.0.0.5:/export rw,vers=4.2,rsize=1048576
41 36 0:40 / /mnt/oldnfs rw,relatime - nfs 10.0.0.6:/export rw,vers=2
42 36 0:41 / /mnt/nfs3 rw,relatime - nfs 10.0.0.7:/export rw,vers=3
44 36 0:44 / /mnt/s3 rw,nosuid,nodev,relatime shared:26 - fuse.s3fs s3fs rw,user_id=0
45 36 0:45 / /mnt/smb rw,relatime - cifs //server/share rw,vers=3.1.1,cache=strict
46 36 0:46 / /mnt/oldsmb rw,relatime - cifs //server/legacy rw,vers=1.0
47 36 0:47 / /mnt/overlay rw,relatime - overlay overlay rw,lowerdir=/a
48 36 0:48 / /mnt/exotic rw,relatime - vendorcsi csi-driver rw
49 36 0:49 / /mnt/with\\040space rw,relatime - xfs /dev/sdc1 rw
"""


def _entry_for(path: str):
    entry = mount_check.find_mount_for(path, mount_check.parse_mountinfo(MOUNTINFO))
    assert entry is not None
    return entry


def test_parses_type_and_options():
    entry = _entry_for("/mnt/nfs/whisperx")
    assert entry.mount_point == "/mnt/nfs"
    assert entry.fs_type == "nfs4"
    assert "vers=4.2" in entry.options


def test_longest_prefix_wins_over_parent_mount():
    # /mnt is ext4 and /mnt/nfs is nfs4; a path under the latter must not be
    # classified using the former.
    assert _entry_for("/mnt/nfs/whisperx/claims").fs_type == "nfs4"
    assert _entry_for("/mnt/whisperx").fs_type == "ext4"


def test_mount_point_octal_escapes_are_decoded():
    assert _entry_for("/mnt/with space/x").mount_point == "/mnt/with space"


@pytest.mark.parametrize(
    "path",
    ["/mnt/nfs/x", "/mnt/nfs3/x", "/mnt/smb/x", "/mnt/overlay/x", "/mnt/x", "/x"],
)
def test_coherent_and_local_types_pass(path):
    assert mount_check.classify(_entry_for(path))[0] == "ok"


@pytest.mark.parametrize("path", ["/mnt/s3/x", "/mnt/oldnfs/x"])
def test_known_broken_types_refuse(path):
    verdict, reason = mount_check.classify(_entry_for(path))
    assert verdict == "refuse"
    assert reason


def test_refusal_names_the_filesystem_type():
    _, reason = mount_check.classify(_entry_for("/mnt/s3/x"))
    assert "fuse.s3fs" in reason


def test_unknown_type_warns_but_does_not_refuse():
    verdict, reason = mount_check.classify(_entry_for("/mnt/exotic/x"))
    assert verdict == "warn"
    assert "vendorcsi" in reason


def test_old_smb_warns_rather_than_refusing():
    assert mount_check.classify(_entry_for("/mnt/oldsmb/x"))[0] == "warn"


def _patch_mountinfo(monkeypatch, tmp_path, text: str) -> str:
    path = tmp_path / "mountinfo"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(mount_check, "_MOUNTINFO_PATH", str(path))
    return str(path)


def test_check_mount_refuses_broken_type(monkeypatch, tmp_path):
    _patch_mountinfo(monkeypatch, tmp_path, MOUNTINFO)
    monkeypatch.setattr(os.path, "realpath", lambda p: "/mnt/s3/whisperx")
    with pytest.raises(StorageCapabilityError) as exc:
        mount_check.check_mount("/mnt/s3/whisperx", allow_unsafe=False)
    assert "fuse.s3fs" in str(exc.value)
    assert "STORAGE__FS__ALLOW_UNSAFE_MOUNT" in str(exc.value)


def test_allow_unsafe_mount_downgrades_refusal_to_warning(monkeypatch, tmp_path):
    _patch_mountinfo(monkeypatch, tmp_path, MOUNTINFO)
    monkeypatch.setattr(os.path, "realpath", lambda p: "/mnt/s3/whisperx")
    mount_check.check_mount("/mnt/s3/whisperx", allow_unsafe=True)


def test_missing_mount_table_is_skipped_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mount_check, "_MOUNTINFO_PATH", str(tmp_path / "does-not-exist")
    )
    mount_check.check_mount(str(tmp_path), allow_unsafe=False)


async def test_store_without_hard_links_fails_the_probe(monkeypatch, tmp_path):
    """The s3fs simulation: link() raising ENOSYS must surface as a capability error."""
    store = FsObjectStore(FsStorageConfig(root=str(tmp_path)))
    await store.open()

    def _no_link(src, dst):
        raise OSError(errno.ENOSYS, "Function not implemented")

    monkeypatch.setattr(os, "link", _no_link)
    with pytest.raises(StorageCapabilityError):
        await store.put_if_absent(key="claims/j1", data=b"{}")
