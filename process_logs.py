from pathlib import Path
import zipfile
import tempfile
import shutil
import re


OUTER_ITER_RE = re.compile(r"Outer Iteration\s+(\d+)")
HEADER_ROW_RE = re.compile(r"^iter,\s*loss")


def parse_actor_log(path: Path):
    outer_iter = None
    data_lines = []
    collecting = False

    for line in path.read_text(errors="replace").splitlines():

        stripped = line.strip()

        m = OUTER_ITER_RE.search(line)
        if m:
            outer_iter = int(m.group(1))

        if HEADER_ROW_RE.match(stripped):
            collecting = True
            continue

        if collecting and stripped:
            data_lines.append(stripped)

    return outer_iter, data_lines


def build_actor_losses_log(run_dir, output_file):

    logs = sorted(
        run_dir.rglob("actor_losses_iter_*.log")
    )

    parsed = []

    for f in logs:

        outer_iter, data = parse_actor_log(f)

        if outer_iter is not None:
            parsed.append((outer_iter, data))

    parsed.sort(key=lambda x: x[0])

    with open(output_file, "w") as out:

        for outer_iter, data in parsed:

            out.write(
                f"=== Outer Iteration {outer_iter} ===\n"
            )

            out.write(
                "iter, loss\n"
            )

            for row in data:
                out.write(row + "\n")

            out.write("\n")


# -----------------------------
# Paths
# -----------------------------

repo_log = (
    Path.home()
    / "Time-consistent_Deephedging"
    / "training_run_logs"
    / "alpha_0.95"
    / "log"
)

modified = repo_log / "modified"
unmodified = repo_log / "unmodified"


# delete old outputs
shutil.rmtree(modified, ignore_errors=True)
shutil.rmtree(unmodified, ignore_errors=True)

modified.mkdir()
unmodified.mkdir()


# -----------------------------
# Save original files
# -----------------------------

shutil.copy2(
    repo_log / "actor_losses.log",
    unmodified / "actor_losses.log"
)

shutil.copy2(
    repo_log / "training.log",
    unmodified / "training.log"
)


# -----------------------------
# Read ZIP
# -----------------------------

zip_path = Path(
    "/mnt/c/Users/paral/Downloads/results (87).zip"
)


with tempfile.TemporaryDirectory() as tmpdir:

    tmp = Path(tmpdir)

    with zipfile.ZipFile(zip_path) as z:

        for name in z.namelist():

            if (
                "run_20260730_141238" in name
                and (
                    "actor_losses_iter_" in name
                    or name.endswith("training.log")
                )
            ):

                target = tmp / Path(name).name

                with z.open(name) as src:
                    with open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)


    # -----------------------------
    # Create modified actor_losses.log
    # -----------------------------

    build_actor_losses_log(
        tmp,
        modified / "actor_losses.log"
    )


    # -----------------------------
    # Copy ZIP training.log
    # -----------------------------

    shutil.copy2(
        tmp / "training.log",
        modified / "training.log"
    )


print("Finished")
print("modified:", modified)
print("unmodified:", unmodified)
