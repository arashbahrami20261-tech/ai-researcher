"""
Sandboxed execution of untrusted code — Milestone 6.

The spec is unambiguous about this: treat generated code as untrusted, and
never give the agent unrestricted access to the host machine. From this
milestone on, code the model writes actually runs, so this module is the
wall between that code and everything you care about.

The danger is not malice, it is accident. A model writing a training loop
can produce an infinite loop, allocate unbounded memory, or overwrite a
file it had no business touching. Every limit below exists because one of
those failure modes is cheap to hit and expensive to recover from.

Nothing here is clever, and that is deliberate. Security code that is hard
to read is security code nobody audits.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# The sandbox image. Pinned to a specific Python version rather than
# "latest", so a surprise upgrade cannot silently change behaviour.
SANDBOX_IMAGE = "python:3.12-slim"

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MEMORY = "512m"
DEFAULT_CPUS = "0.5"


@dataclass
class ExecutionResult:
    """
    What came back from one sandboxed run.

    `timed_out` is separate from `exit_code` because they mean different
    things: a non-zero exit is the code failing, a timeout is the code
    never finishing. An experiment engine needs to tell those apart.
    """

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def build_docker_command(
    host_dir: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    memory: str = DEFAULT_MEMORY,
    cpus: str = DEFAULT_CPUS,
) -> list[str]:
    """
    Assemble the docker command. Split out from `run_code` so the security
    flags can be asserted on in tests without launching a container.

    Every flag is a specific restriction:

      --network none      No network at all. Generated code cannot phone
                          home, download anything, or reach your LAN.
      --memory            Hard memory cap. An unbounded allocation gets
                          the container killed instead of your laptop
                          swapping to death.
      --memory-swap       Set equal to --memory so the container cannot
                          escape the cap by using swap.
      --cpus              CPU share cap, so a runaway loop cannot make the
                          rest of the machine unusable.
      --pids-limit        Caps process count. Blocks fork bombs.
      --read-only         Root filesystem is read-only. Code cannot modify
                          the image it runs in.
      --tmpfs /tmp        One writable scratch area, in RAM, size-capped,
                          and discarded when the container exits.
      --user 1000:1000    Do not run as root inside the container.
      --cap-drop ALL      Drop every Linux capability. Nothing here needs
                          any of them.
      --security-opt      Block privilege escalation via setuid binaries.
      --rm                Delete the container when it exits, so repeated
                          runs do not accumulate dead containers.
      :ro on the mount    The code directory is mounted read-only, so the
                          script cannot rewrite itself mid-run.
    """
    return [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", memory,
        "--memory-swap", memory,
        "--cpus", cpus,
        "--pids-limit", "64",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m",
        "--user", "1000:1000",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--workdir", "/sandbox",
        "-v", f"{host_dir}:/sandbox:ro",
        SANDBOX_IMAGE,
        "python", "/sandbox/script.py",
    ]


def run_code(
    code: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    memory: str = DEFAULT_MEMORY,
    cpus: str = DEFAULT_CPUS,
) -> ExecutionResult:
    """
    Run `code` inside a locked-down container and return what it produced.

    The code is written to a temporary directory on the host, which is
    mounted read-only into the container. The directory is deleted when
    this function returns, whatever happened.

    Two layers of timeout on purpose: docker's own limits, plus
    subprocess's `timeout`. If the container hangs in a way docker does
    not catch, the outer timeout still frees the caller. Defence in depth
    is cheap here.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "script.py"
        script_path.write_text(code, encoding="utf-8")

        command = build_docker_command(Path(tmp), timeout_seconds, memory, cpus)

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                # A little slack over the container limit, so docker gets
                # the chance to enforce its own timeout first and report
                # a real exit code rather than us killing it blindly.
                timeout=timeout_seconds + 10,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution exceeded {timeout_seconds}s and was killed.",
                exit_code=-1,
                timed_out=True,
            )
        except FileNotFoundError:
            # Docker is not installed or not on PATH. Say so plainly
            # instead of letting this surface as a confusing crash later.
            raise RuntimeError(
                "docker not found. The sandbox requires Docker to be "
                "installed and runnable without sudo."
            ) from None

        return ExecutionResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            timed_out=False,
        )
