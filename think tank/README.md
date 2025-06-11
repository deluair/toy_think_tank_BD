# Bangladesh AI Think Tank – Open Blueprint

This repository contains the datasets, Python scripts, and documentation for an AI-powered policy think-tank concept tailored to Bangladesh.

## Project Structure

```
.
├── assets/           # Images, logos, and other static assets
├── data/             # CSV datasets
├── docs/             # Detailed documentation and markdown files
├── src/              # Python source code
│   ├── main.py       # Main script to generate data and plots
│   ├── ...           # Other scripts and modules
├── .gitignore
├── README.md         # You are here
├── requirements.txt  # Python dependencies
```

## Quick Start

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/deluair/toy_think_tank_BD.git
    cd toy_think_tank_BD
    ```

2.  **Set up a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the main script**:

    Navigate to the source directory and run the main script:
    ```bash
    python src/main.py
    ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

- [ ] Add automated tests (pytest)
- [ ] Set up GitHub Actions for CI/CD
- [ ] Develop interactive visualizations
- [ ] Containerize the application with Docker

## License

All content is released under the permissive **MIT License** unless noted otherwise.
