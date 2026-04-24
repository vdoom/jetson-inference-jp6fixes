# Building jetson-inference on Xavier NX (JetPack 5) for Python 3.10

JetPack 5 on Xavier NX runs Ubuntu 20.04 (focal), whose system Python is 3.8.
This branch (`XNX_Py3.10`) lets you build the C++ libraries and the Python
bindings against an **altinstalled Python 3.10** inside a venv, without
rebuilding Python as a shared library.

## What this branch changes

1. `python/CMakeLists.txt` and `utils/python/CMakeLists.txt` — the `focal`
   branch of the codename matrix now targets Python **3.10** instead of 3.8.
2. `python/bindings/CMakeLists.txt` and `utils/python/bindings/CMakeLists.txt`:
   - Use `find_package(Python3 ... COMPONENTS Interpreter NumPy)`; do **not**
     request `Development`, so `libpython3.10.so` is not required.
   - Do **not** link `libpython` into the binding `.so`. On Linux, Python
     extension modules resolve `Py*` symbols at `dlopen()` time from the
     host interpreter, which means an altinstalled Python (configured
     without `--enable-shared`, i.e. only `libpython3.10.a` present) is fine.
   - When `VIRTUAL_ENV` is set at configure time, the bindings install
     directly into `$VIRTUAL_ENV/lib/python3.10/site-packages/`.
   - `-DPYTHON_BINDING_INSTALL_DIR=/some/path` overrides the install path.

## Prerequisites

- Xavier NX flashed with JetPack 5 (Ubuntu 20.04 / focal).
- CUDA, cuDNN, TensorRT from JetPack (already present on a standard JP5 flash).
- Python 3.10 installed via `make altinstall` to `/usr/local` (headers at
  `/usr/local/include/python3.10/Python.h`). Building Python with
  `--enable-shared` is **not** required.
- A Python 3.10 virtualenv. Example:
  ```bash
  python3.10 -m venv ~/venv310
  source ~/venv310/bin/activate
  pip install --upgrade pip wheel
  pip install numpy
  ```

## Build

```bash
source ~/venv310/bin/activate

git clone <your-fork-url> jetson-inference
cd jetson-inference
git checkout XNX_Py3.10
git submodule update --init --recursive
# make sure the utils submodule is on its XNX_Py3.10 branch:
git -C utils checkout XNX_Py3.10

mkdir build && cd build

# IMPORTANT: run cmake as your user, NOT under sudo.
# sudo strips VIRTUAL_ENV and the install path will fall back
# to /usr/lib/python3.10/dist-packages/.
cmake ../ \
    -DBUILD_INTERACTIVE=NO \
    -DPython3_EXECUTABLE=$VIRTUAL_ENV/bin/python3.10 \
    -DPython3_FIND_VIRTUALENV=ONLY
```

Verify the cmake output contains:

```
-- found Python version:  3.10
-- found Python interp:   /home/<user>/venv310/bin/python3.10
-- found Python include:  /usr/local/include/python3.10
-- found NumPy version:   1.22.4   (or similar)
-- installing Python bindings into active venv: /home/<user>/venv310/lib/python3.10/site-packages
```

Then:

```bash
make -j$(nproc)
sudo make install     # sudo only for the install step; paths are already baked in
sudo ldconfig
```

`BUILD_INTERACTIVE=NO` skips `install-pytorch.sh`, which otherwise tries to
install NVIDIA's JP5 PyTorch wheel built for system Python 3.8 — that wheel
will not work in a 3.10 venv.

## Python dependencies

The pure-Python side of `jetson_utils` imports a few packages that are not
declared anywhere. Install them into the venv:

```bash
pip install requests termcolor tabulate docker numpy
```

## Verify

```bash
source ~/venv310/bin/activate
python -c "import jetson_utils, jetson_inference; print('ok')"
```

Should print `ok`.

## Troubleshooting

### Bindings went to `/usr/lib/python3.10/dist-packages/` instead of the venv

You ran `cmake` under `sudo`. Re-run cmake as your user with the venv
activated, then `sudo make install`.

As a one-shot fix without reconfiguring, symlink the installed files into
the venv:

```bash
SITE=$VIRTUAL_ENV/lib/python3.10/site-packages
SRC=/usr/lib/python3.10/dist-packages
for n in jetson_utils_python.so jetson_inference_python.so \
         jetson_utils jetson_inference jetson Jetson; do
    ln -sfn "$SRC/$n" "$SITE/$n"
done
```

### `ModuleNotFoundError: No module named 'termcolor' / 'tabulate' / 'requests'`

Pure-Python deps not installed in the venv. See the *Python dependencies*
section above.

### `ImportError: ... undefined symbol: Py...`

The extension module was linked against a different libpython, or a stray
libpython got pulled in by cmake. Confirm:

```bash
ldd $VIRTUAL_ENV/lib/python3.10/site-packages/jetson_utils_python.so | grep -i python
```

It should show **no** `libpython*` dependency. If it does, you're building
against a modified source tree that still has `Python3::Python` /
`${PYTHON_LIBRARIES}` in `target_link_libraries`.

### `tensorrt` import fails from the 3.10 venv

JetPack 5 ships the `tensorrt` Python bindings only for system Python 3.8.
`jetson-inference` itself uses TensorRT through its **C++** API, so the
library works fine in a 3.10 venv. If downstream Python code needs
`import tensorrt`, you have to build TRT Python bindings for 3.10 from the
TensorRT OSS source — that is out of scope for this branch.

## Background: why don't we link libpython?

A CPython extension module is loaded by an already-running interpreter via
`dlopen()` (plus `PyModule_Create`). At that moment the interpreter's
`Py*` symbols are already present in the process's symbol table, so they
resolve automatically. Linking libpython into the extension is not required,
and on CPython builds where only a static `libpython.a` exists, linking it
is actively harmful (duplicate interpreter state, PIC relocation errors).
This branch drops that link, which is the conventional Linux-extension
build recipe.
