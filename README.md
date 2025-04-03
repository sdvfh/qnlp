# Quantum Natural Language Processing

This repository contains the implementation and experimental framework developed for my Master's Degree final project. The project investigates the application of quantum computing techniques to natural language processing (QNLP), integrating quantum algorithms with classical methodologies to analyze and process textual data.

## Prerequisites

- Python 3.12.8
- All required dependencies are listed in the `requirements.txt` file.

To install the necessary dependencies, execute the following command:

    pip install -r requirements.txt

## How to Run

1. Clone the repository:

       git clone https://github.com/sdvfh/qnlp.git

2. Navigate to the `code` directory:

       cd qnlp/code

3. Set up the datasets:
   
       python setup_datasets.py

4. Execute the main experiment script:
   ```bash
   export TOKENIZERS_PARALLELISM=false
   python experiments.py
   ```

## Project Structure

The repository is organized as follows:

    .
    ├── code/
    │   └── experiment.py
    ├── requirements.txt
    └── README.md

## License

This project is distributed under the terms of the [MIT License](LICENSE).
