"""
Tests for security/sandbox.py.

Split in two. The offline tests assert on the docker command we build,
without launching anything — they run in CI, where spinning up containers
is slow and awkward.

The tests marked `live` actually execute code in a real container. Those
are the ones that matter: a security control nobody has watched fail is
not a control, it is a hope. Run them with:  pytest -m live
"""

from __future__ import annotations

import pathlib

import pytest

from security.sandbox import DEFAULT_MEMORY, build_docker_command, run_code


# ---------- offline: assert on the command we construct ----------

def _flags():
    return build_docker_command(pathlib.Path("/tmp/whatever"))


def test_network_is_disabled():
    cmd = _flags()
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "none"


def test_memory_is_capped_and_swap_cannot_bypass_it():
    cmd = _flags()
    memory = cmd[cmd.index("--memory") + 1]
    swap = cmd[cmd.index("--memory-swap") + 1]
    # Equal values mean the container gets no swap beyond its RAM cap.
    # Without this, a memory limit is trivially escaped.
    assert memory == DEFAULT_MEMORY
    assert swap == memory


def test_filesystem_is_read_only_and_code_mount_is_too():
    cmd = _flags()
    assert "--read-only" in cmd
    mount = cmd[cmd.index("-v") + 1]
    assert mount.endswith(":ro")


def test_container_does_not_run_as_root():
    cmd = _flags()
    assert cmd[cmd.index("--user") + 1] != "0:0"


def test_capabilities_are_dropped_and_escalation_blocked():
    cmd = _flags()
    assert cmd[cmd.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in cmd


def test_process_count_is_capped():
    # Without a pids limit, a three-line fork bomb takes down the host.
    cmd = _flags()
    assert "--pids-limit" in cmd


def test_container_is_removed_after_running():
    assert "--rm" in _flags()


# ---------- live: actually run code in a container ----------

@pytest.mark.live
def test_plain_code_runs_and_returns_output():
    result = run_code("print(2 + 2)")
    assert result.succeeded
    assert result.stdout.strip() == "4"


@pytest.mark.live
def test_network_access_is_blocked_in_practice():
    result = run_code(
        "import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)"
    )
    assert not result.succeeded


@pytest.mark.live
def test_host_files_are_not_visible():
    result = run_code("import os; print(os.listdir('/home'))")
    # The container has its own filesystem; the host's /home is not in it.
    assert "arash" not in result.stdout


@pytest.mark.live
def test_writing_outside_tmp_fails():
    result = run_code("open('/evil.txt', 'w').write('pwned')")
    assert not result.succeeded


@pytest.mark.live
def test_infinite_loop_is_killed_by_the_timeout():
    result = run_code("while True: pass", timeout_seconds=5)
    assert not result.succeeded


@pytest.mark.live
def test_failing_code_reports_its_error_rather_than_vanishing():
    result = run_code("raise ValueError('boom')")
    assert not result.succeeded
    assert "boom" in result.stderr
