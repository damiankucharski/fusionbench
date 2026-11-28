# Reproduce a packaged reference bank

The working v1 includes 30 fixed banks in six collections. Each bank contains 128 training and 128 test examples generated with seed 1729. Use `load_reference_dataset(scenario_id)` to load its original labels, provider probabilities, oracles and provenance offline. The loader checks the collection manifest and the archive digest before reading the arrays.

| Collection | Banks added | Original source and build script |
| --- | ---: | --- |
| reference-v1 | 19 | [e51f2e7](https://github.com/damiankucharski/fusionbench/blob/e51f2e71f556c8165306bfe18deb664cdb4ca299/tools/build_reference.py) |
| reference-v2 | 1 | [da4492a](https://github.com/damiankucharski/fusionbench/blob/da4492ab36c1b6ddd07cd81b88b6b68512719531/tools/build_reference.py) |
| reference-v3 | 4 | [a218ee1](https://github.com/damiankucharski/fusionbench/blob/a218ee1071009a5fa3076bb49d3f40e06e6a8d6c/tools/build_reference.py) |
| reference-v4 | 1 | [428eb4a](https://github.com/damiankucharski/fusionbench/blob/428eb4abb1777f9f07d9a365fe642164701a347f/tools/build_reference.py) |
| reference-v5 | 2 | [06dae84](https://github.com/damiankucharski/fusionbench/blob/06dae84ea915910ec496a9f8d86976d0aba7a82a/tools/build_reference.py) |
| reference-v6 | 3 | [aaac530](https://github.com/damiankucharski/fusionbench/blob/aaac530c80cef246c27cc710fa8bbb8ff0a35688/tools/build_reference.py) |

The historical archives include the generator version and the parameters in effect at build time. To reproduce the files including their headers, use the original revision, its lockfile and the build environment (Linux x86-64, Python 3.12.14, NumPy 2.5.0, SciPy 1.18.0). For example, from a clone with the research branch available:

```sh
git worktree add --detach ../fusionbench-reference-v6 aaac530c80cef246c27cc710fa8bbb8ff0a35688
cd ../fusionbench-reference-v6
uv sync --locked --python 3.12.14
uv run --no-sync python tools/build_reference.py --version reference-v6 --out /tmp/rebuilt-v6
diff -rq fusionbench/data/reference-v6 /tmp/rebuilt-v6
```

The original reference-v1 script accepts only `--out`. For a new data seed or different generator settings, use `generate_dataset()` and `save_dataset()` as in the [dataset guide](datasets.md). A new dataset records the version used to generate it.
