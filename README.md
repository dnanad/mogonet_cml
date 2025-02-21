# Multi-Omic Data Analysis Pipeline with MOGONET

This repository contains a comprehensive pipeline for multi-omic data analysis, built upon the MOGONET framework. The pipeline includes preprocessing, classical machine learning model evaluation, and result aggregation to facilitate comparative analysis.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Pipeline Overview](#pipeline-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Multi-omic data analysis involves integrating and analyzing multiple types of omics data (e.g., cfRNA, metabolomics) to gain a holistic understanding of biological systems. This pipeline leverages the MOGONET framework to apply Graph Neural Networks (GNNs) for this purpose and includes additional components for preprocessing, classical ML model evaluation, and result aggregation.

## Features

- **Feature Selection and Preprocessing**: Automated feature selection and data preprocessing to ensure high-quality input for models.
- **Classical Machine Learning Models**: Baseline models for performance comparison.
- **Graph Neural Networks with MOGONET**: Advanced modeling using GNNs for multi-omic data.
- **Result Aggregation**: Comprehensive aggregation and comparison of model results.

## Pipeline Overview

The pipeline consists of the following main scripts:

1. **main_preprocessing.py**: Handles feature selection, data preprocessing, and train-test splitting.
2. **main_cml.py**: Implements classical machine learning models for baseline performance.
3. **main_mogonet.py**: Core script for training the MOGONET model.
4. **main_result_aggregator.py**: Aggregates and compares results from classical ML and MOGONET models.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dnanad/mogonet_cml.git
   cd mogonet_cml
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage



## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact [ananddeshpande13@gmail.com](mailto:ananddeshpande13@gmail.com).