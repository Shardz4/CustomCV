# Contributing to CustomCV

Thank you for your interest in contributing to **CustomCV**! We welcome and appreciate contributions of all kinds, whether you are fixing bugs, adding new computer vision algorithms, improving documentation, or submitting feedback.

CustomCV is a hybrid project with a high-performance computer vision backend written in **Rust** and pythonic bindings generated using **PyO3** and **Maturin**.

---

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Welcome Bots & Automation](#welcome-bots--automation)
3. [Repository Structure](#repository-structure)
4. [Setting Up Your Development Environment](#setting-up-your-development-environment)
5. [Development & Coding Workflow](#development--coding-workflow)
6. [Testing, Linting, & Formatting](#testing-linting--formatting)
7. [Submitting a Pull Request](#submitting-a-pull-request)

---

## Code of Conduct

We aim to foster an open, welcoming, and inclusive community. Please be respectful and constructive in all interactions, including issues, pull requests, and discussions.

---

## Welcome Bots & Automation

To help welcome new contributors and streamline triage, this repository uses the [GitHub Welcome App](https://github.com/apps/welcome). You will interact with this bot at various stages of your contribution:

1. **First-time Issues**: When you open your first issue, the welcome bot will reply with a greeting and ask you to ensure that you have included steps to reproduce the problem along with relevant environment details (OS, Rust version, and Python version).
2. **First-time Pull Requests**: When you open your first pull request, the welcome bot will point you to this guide and kindly ask you to verify that CI is passing and that you have run `cargo clippy` and `cargo test` locally.
3. **First-time Merges**: Once your first pull request is merged, the welcome bot will post a celebratory comment welcoming you as an official contributor!

---

## Repository Structure

Before diving in, it is helpful to understand how the codebase is laid out:

* [src/](file:///c:/Users/CREWMOBILE/Desktop/dip_lab/collated_library/collection/src): Contains the Rust source files.
  * [src/lib.rs](file:///c:/Users/CREWMOBILE/Desktop/dip_lab/collated_library/collection/src/lib.rs): The PyO3 module entry point where all Rust functions are registered and exposed to Python.
  * [src/helpers.rs](file:///c:/Users/CREWMOBILE/Desktop/dip_lab/collated_library/collection/src/helpers.rs): Internal Rust utility functions (e.g., kernels, helper algorithms) not directly exposed to Python.
  * Other modules (e.g., `transforms.rs`, `color_convert.rs`, `edge_detection.rs`): Implementations of specific groups of computer vision operations.
* [Cargo.toml](file:///c:/Users/CREWMOBILE/Desktop/dip_lab/collated_library/collection/Cargo.toml): Rust package configuration. Specifies `crate-type = ["cdylib"]` so it can be compiled into a Python extension.
* [pyproject.toml](file:///c:/Users/CREWMOBILE/Desktop/dip_lab/collated_library/collection/pyproject.toml): Python build system configuration using `maturin` as the build backend.
* [.github/workflows/CI.yml](file:///c:/Users/CREWMOBILE/Desktop/dip_lab/collated_library/collection/.github/workflows/CI.yml): GitHub Actions workflow that builds and tests the library wheels across Linux, macOS, and Windows.

---

## Setting Up Your Development Environment

To compile and run CustomCV locally, you will need both the Rust toolchain and Python.

### Prerequisites
1. **Rust**: Install Rust via [rustup](https://rustup.rs/).
2. **Python**: Python 3.8 or newer.
3. **Maturin**: The build tool that links Rust PyO3 code with Python.

### Step-by-Step Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Shardz4/CustomCV.git
   cd CustomCV/collection
   ```

2. **Create and Activate a Virtual Environment**:
   It is highly recommended to use a virtual environment to isolate dependencies.
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Build Tools**:
   Install `maturin` and any other required Python utilities:
   ```bash
   pip install --upgrade pip
   pip install maturin
   ```

4. **Build and Install in Development/Editable Mode**:
   Use Maturin to compile the Rust library and install it directly into your virtual environment:
   ```bash
   maturin develop
   ```
   This compiles the extension and makes it importable as `import rust_cv_lib` inside Python. If you make changes to the Rust code, you must rerun `maturin develop` to recompile the library.

---

## Development & Coding Workflow

If you want to add or modify a computer vision function:

1. **Identify the module**: Find the appropriate file in the `src/` directory (e.g., `src/filters.rs` for spatial filtering, `src/color_convert.rs` for color spaces).
2. **Implement the function**: Write the Rust code. If it uses numpy arrays, leverage the `numpy` and `ndarray` crates (check existing functions in the codebase for reference).
3. **Register the function**:
   Open [src/lib.rs](file:///c:/Users/CREWMOBILE/Desktop/dip_lab/collated_library/collection/src/lib.rs) and ensure your function is registered in the PyO3 module block:
   ```rust
   #[pymodule]
   fn rust_cv_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
       m.add_function(wrap_pyfunction!(your_new_function, m)?)?;
       Ok(())
   }
   ```
4. **Document the function**: Add a short description and update the [README.md](file:///c:/Users/CREWMOBILE/Desktop/dip_lab/collated_library/collection/README.md) function reference table.

---

## Testing, Linting, & Formatting

Maintaining code quality is essential. Please run the following checks locally before submitting your code:

### 1. Formatting Rust Code
Ensure your code adheres to standard Rust formatting guidelines:
```bash
cargo fmt --all -- --check
```
To automatically apply formatting:
```bash
cargo fmt --all
```

### 2. Linting Rust Code
Run Clippy to catch common mistakes and idiomatic issues:
```bash
cargo clippy --all-targets -- -D warnings
```

### 3. Running Rust Tests
Run the project's test suite to ensure no regressions:
```bash
cargo test
```

### 4. Testing Python Interoperability
You can write quick Python verification scripts to import `rust_cv_lib` and call your new functions on test images.
```python
import rust_cv_lib
import numpy as np

# Create a dummy image
img = np.zeros((100, 100), dtype=np.uint8)
# Call your function
result = rust_cv_lib.apply_negative(img)
assert result[0, 0] == 255
print("Success!")
```

---

## Submitting a Pull Request

Once you have implemented your feature or bug fix and verified it locally:

1. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Commit your changes** with descriptive commit messages.
3. **Push to your fork** or branch on GitHub.
4. **Open a Pull Request (PR)** against the `main` branch.
   - Describe the changes and the problem they solve.
   - Link any related issues.
   - Ensure the welcome bot processes your PR and double check that all GitHub Actions CI checks pass.
5. **Request a review** from the maintainers. Once approved and CI is green, your changes will be merged!
