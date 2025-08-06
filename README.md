# Description
This is a Python code for forward acceleration calculation, which combines the finite difference principle and the convolution calculation principle to greatly improve the calculation speed of forward calculation, and the imaging effect is the same as that of finite difference.

# Result
**P-wave field comparison results**.

<img width="1000" height="600" alt="result" src="https://github.com/user-attachments/assets/6357435e-110c-48f1-8e66-4ddd7332d7f2" />

**S-wave field comparison results**.

<img width="1000" height="600" alt="result_s" src="https://github.com/user-attachments/assets/a457b838-ca45-4772-b77e-cd1f2a14d556" />

**Comparison of calculation speed at 4000 sampling points (4 s)**.

<img width="231" height="120" alt="speed" src="https://github.com/user-attachments/assets/b3e8ea33-777d-4c69-be27-32187de6e4a6" />

# Requirements
Hardware requirements: 
- NVIDIA 30 series 8G memory graphics card

Software required:
- python:3.9.19,
- torch:2.0.1+cu118,
- numpy:1.26.3,
- scipy:1.13.0,
- matplotlib:3.9.2
# Usage
Download this code to your folder, configure the appropriate environment, and then execute forward.py.
- python forward.py
# Acknowledgments
This work was supported by Award Project of Heilongjiang Province Science and Technology Innovation Base in China"Key Technology Research on Digital Intelligent Oilfield Information Perception and Intelligent Analysis Processing" (JD24A009).
# License
forwardpy is licensed under the [Apache License 2.0](LICENSE).
