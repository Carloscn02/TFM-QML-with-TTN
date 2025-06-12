# TFM-QML-with-TTN
# TFM-QML-with-TTN

# Quantum MNIST Classifier

This repository contains code and experiments for quantum machine learning applied to the MNIST dataset, using amplitude encoding and custom quantum circuit architectures. The project explores different quantum circuit designs and optimization strategies for binary classification (digits 0 and 1) using Qiskit.

MSNIT datavese in csv format can be found at https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

## Features

- **Data Preprocessing:**  
  - Resizing MNIST images to 4x4, 16x16, and 28x28.
  - Normalization by L2 norm or pixel scaling.
  - Balanced sampling for binary classification.

- **Quantum Circuit Architectures:**  
  - Amplitude encoding of image data into quantum states.
  - Custom parameterized unitary gates built from Hermitian matrices.
  - Flexible circuit depth and connectivity.

- **Training and Optimization:**  
  - SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer for variational circuits.
  - Cost and accuracy tracking during training.
  - Saving and loading of trained parameters.

- **Visualization:**  
  - Plotting of input images, training curves, and quantum measurement histograms.
  - Conversion of quantum measurement results to image-like arrays.

## Folder Structure

All relevant files are in the `MNIST/` folder:

```
MNIST/
├── comparacion.ipynb
├── mnist_test.csv
├── mnist_train.csv
├── test.ipynb
├── V1_16x16_amplitude.ipynb
├── V1_16x16_amplitude(accuracy).txt
├── V1_16x16_amplitude(new_theta).txt
├── V1_16x16_amplitude(Thetalog).txt
├── V1_4x4_amplitude.ipynb
├── V1_4x4_amplitude(accuracy).txt
├── V1_4x4_amplitude(new_theta).txt
├── V1_4x4_amplitude(Thetalog).txt
├── V1_4x4_base.ipynb
├── V1_4x4_base(accuracy).txt
├── V1_4x4_base(new_theta).txt
├── V1_4x4_base(Thetalog).txt
├── V2_16x16_amplitude.ipynb
...
```

- `V1_4x4_amplitude.ipynb`, `V1_16x16_amplitude.ipynb`, `V2_16x16_amplitude.ipynb`: Main quantum classifier experiments with different architectures and image sizes.
- `comparacion.ipynb`, `test.ipynb`: Additional experiments and utilities.
- `*.txt`: Saved results (parameters, accuracy, logs) from experiments.

## Requirements

- Python 3.8+
- [Qiskit](https://qiskit.org/)
- numpy
- pandas
- matplotlib
- scikit-image

Install requirements with:

```bash
pip install qiskit numpy pandas matplotlib scikit-image
```

## Usage

1. **Prepare the MNIST data:**  
   Use only digits 0 and 1. Resize and normalize as needed.

2. **Run the notebooks:**  
   Open any of the provided `.ipynb` files in Jupyter or VS Code and execute the cells to train and evaluate quantum classifiers.

3. **Train a model:**  
   - Set hyperparameters (number of shots, learning rates, etc.).
   - Run the SPSA optimizer.
   - Monitor cost and accuracy plots.

4. **Visualize results:**  
   - Use the provided plotting functions to visualize input images, quantum circuit diagrams, and measurement histograms.

## Example: Running a Quantum Classifier

```python
# Load and preprocess data
df = pd.read_csv('mnist_test.csv', header=None)
df = df[(df[0] == 0) | (df[0] == 1)]
df_4x4 = reduce_to_nxn(df, 4)
df_4x4 = normalize_by_norm(df_4x4)

# Build and run a quantum circuit
Tree = get_tree_V1(np.random.rand(48), 3)
qc = get_circuit(df_4x4.iloc[0, 1:].values, Tree)
qc.draw('mpl')
```

## Citation

If you use this code for your research, please cite this repository.

## License

This project is licensed under the MIT License.

---

**Author:**  
Carlos Cabezas Navarro 
