# Hybrid Neural Network with Adversarial Defense Choices

This repository contains code for a Hybrid Neural Network (HNN) that combines a Quantum Neural Network (QNN) and a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST or EMNIST datasets. The HNN model is protected against adversarial attacks using either adversarial training or randomization defense techniques.

## Motivating Articles and Related Works

Sitawarin, C., Golan-Strieb, Z.J. &amp; Wagner, D.. (2022). Demystifying the Adversarial Robustness of Random Transformation Defenses. <i>Proceedings of the 39th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 162:20232-20252 Available from https://proceedings.mlr.press/v162/sitawarin22a.html

Sicong Han, Chenhao Lin, Chao Shen, Qian Wang, and Xiaohong Guan. 2023. Interpreting Adversarial Examples in Deep Learning: A Review. ACM Comput. Surv. 55, 14s, Article 328 (December 2023), 38 pages. https://doi.org/10.1145/3594869

Olga Taran, Shideh Rezaeifar, Taras Holotyak, Slava Voloshynovskiy; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 11226-11233
https://openaccess.thecvf.com/content_CVPR_2019/html/Taran_Defending_Against_Adversarial_Attacks_by_Randomized_Diversification_CVPR_2019_paper.html

## Results

### Adversarial Training Defense Mechanism for HNN Model with MNIST Dataset

| **Compounded Attack** | **Pre-Attack Accuracy** | **Post Attack Accuracy** | **Post Attack Defense Applied Accuracy** |
|------------------|------------------|------------------|--------------------|
| FGSM + CW        | 98.0%           | 20.0%           | 100.0%             |
| FGSM + PGD       | 98.0%           | 20.0%           | 98.0%             |
| CW + PGD         | 100.0%           | 89.0%           | 100.0%             |


## Features

- Hybrid Neural Network (HNN) architecture combining QNN and CNN
- Support for MNIST and EMNIST datasets
- Adversarial defense using either adversarial training or randomization techniques
- Multiple randomization defense methods:
  - Random resizing
  - Random cropping
  - Random rotation
  - Combined randomization (resizing, cropping, rotation, color jittering, random erasing, noise injection)
- Evaluation of model performance and adversarial robustness
- Visualization of misclassified examples and adversarial perturbations

## Adversarial Defense Techniques

### Adversarial Training

Adversarial training is a defense technique that involves training the model on a combination of clean examples and adversarial examples generated using various attack methods. By exposing the model to adversarial examples during training, it learns to be more robust and resistant to adversarial perturbations.

### Randomization Defense

Randomization defense techniques involve applying random transformations to the input data during training to increase the model's robustness against adversarial attacks. The key idea behind randomization defense is to introduce random variations in the input data, making it harder for adversarial perturbations to have a consistent effect on the model's predictions.

The implemented randomization defense methods work as follows:

- Random Resizing: The input images are randomly resized within a specified scale range. This introduces variations in the spatial dimensions of the images, making the model more resilient to size-related adversarial perturbations.
- Random Cropping: A random smaller region is cropped from the input images. This helps the model learn to focus on different parts of the image and reduces its sensitivity to specific pixel locations.
- Random Rotation: The input images are randomly rotated within a specified angle range. This helps the model become invariant to rotational changes and enhances its ability to recognize objects from different orientations.
- Combined Randomization: Multiple randomization techniques, including resizing, cropping, rotation, color jittering, random erasing, and noise injection, are applied together. This creates a diverse set of input variations, making it challenging for adversarial perturbations to have a consistent impact.

By applying these randomization techniques, the model learns to be more robust and generalizable, as it is trained on a wide range of input variations. Adversarial perturbations that are crafted based on a specific input may not have the same effect when random transformations are applied, reducing the effectiveness of adversarial attacks.

## Importance of Model Protection

Protecting machine learning models throughout the development phase is crucial to ensure their security and reliability. Adversarial attacks can easily fool unprotected models, leading to incorrect predictions and potentially harmful consequences in real-world applications.

By implementing adversarial defense techniques, such as adversarial training and randomization defense, we can enhance the model's robustness and reduce its vulnerability to adversarial attacks. This helps to maintain the integrity and trustworthiness of the model's predictions.

## Adversarial Attack Scenarios

In this code, we consider a white-box, targeted, compounded adversarial attack scenario. In a white-box attack, the adversary has full knowledge of the model's architecture, parameters, and training data. Targeted attacks aim to cause the model to misclassify examples into specific target classes. Compounded attacks involve a combination of multiple attack methods to create more sophisticated and challenging adversarial examples.

Adversaries may attempt to attack machine learning models for various reasons, such as:

- Malicious Intent: Adversaries may seek to cause harm or disrupt the model's functionality by inducing incorrect predictions.
- Fooling the Model: Attackers may aim to deceive the model into making wrong classifications or decisions for their own benefit.
- Exploiting Vulnerabilities: Adversaries may attempt to identify and exploit vulnerabilities in the model to compromise its security or privacy.

By protecting the model against such advanced adversarial attacks, we can enhance its robustness and mitigate the risks associated with adversarial vulnerabilities.

## Usage

To use the code in this repository:

1. Install the required dependencies (PyTorch, torchvision, torchattacks, NumPy, tabulate).
2. Set the desired dataset (`dataset_name`) and compounded attack method (`compounded_attack_name`).
3. Choose the defense type (`defense_type`) and randomization defense method (`randomization_defense`) if applicable.
4. Run the code to train the HNN model, evaluate its performance, and assess its robustness against adversarial attacks.

Please refer to the code documentation and comments for more details on the implementation and usage.

## Acknowledgements

This code builds upon the concepts and techniques from various research papers and open-source libraries in the field of adversarial machine learning. We would like to acknowledge their contributions and the valuable insights they provide.

Disclaimer This repository is intended for educational and research purposes.
