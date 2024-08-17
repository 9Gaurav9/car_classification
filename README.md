
# Fine-Grained Car Model Classification

## Overview

This project focuses on classifying car images into specific makes, models, and years using deep learning techniques. The system is built with PyTorch and involves a custom model architecture for accurate classification.

## Project Structure

- **`car_classification/`**: Contains the core functionality of the project.
  - **`model.py`**: Defines the model architecture.
  - **`data.py`**: Handles data loading and preprocessing.
  - **`train.py`**: Script for training the model.
  - **`predict.py`**: Script for evaluating the model on test data.
- **`check.py`**: A utility script for checking the setup and dependencies.
- **`requirements.txt`**: Lists the Python dependencies required for the project.

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/car_classification.git
   cd car_classification
   ```

2. **Create and Activate Virtual Environment:**

   ```bash
   python -m venv myenv
   myenv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download and Prepare Data:**

   - Place your car dataset in a directory.
   - Update the `train_dir` and `test_dir` paths in the `predict.py` and `train.py` files to point to your dataset.

## Usage

### Training the Model

To train the model, run:

```bash
python train.py
```

### Evaluating the Model

To evaluate the model's performance, run:

```bash
python predict.py
```

### Checking the Setup

To check the setup and dependencies, run:

```bash
python check.py
```

## Model Architecture

The model is defined in `car_classification/model.py` and utilizes a custom architecture for fine-grained classification of car images.

## Project Goals

- Develop a robust model to classify car images into specific makes, models, and years.
- Implement efficient data handling and preprocessing techniques.
- Achieve high classification accuracy on test data.

## Contributing

Feel free to submit issues or pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Stanford University AI Lab Car Dataset.
- Deep learning libraries and tools: PyTorch.


DataSet Citation
Dataset - https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
3D Object Representations for Fine-Grained Categorization

Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei

4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013
