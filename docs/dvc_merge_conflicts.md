# Handling merge conflicts with DVC

DVC keeps large data out of Git. Git only stores a small pointer file
(`data/raw/paysim.csv.dvc`) holding the data's MD5 hash and size.

## What conflicts look like
If two branches each re-add the dataset, both change the `md5` and `size` lines
of `paysim.csv.dvc`, and Git reports a conflict in that file. The data itself
never conflicts, because it is not in Git. `dvc.lock` can conflict the same way
when two branches rerun a pipeline stage.

## How we resolve it
1. Decide which dataset version should win, or whether a combined dataset is needed.
2. Keep that side of the pointer file:
   `git checkout --ours data/raw/paysim.csv.dvc` (or `--theirs`), or edit the file by hand.
3. `git add data/raw/paysim.csv.dvc`, then finish the merge commit.
4. Run `dvc checkout` so the workspace file matches the pointer.
5. If a combined dataset is needed, rebuild it, run `dvc add`, commit and `dvc push`.
6. For `dvc.lock`: take either side, then run `dvc repro` to regenerate it.

Never resolve a conflict by editing the data file inside Git, and never commit the
real CSV.

Note: this procedure follows DVC's documented behavior. We did not stage a conflict
in this project.