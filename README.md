# Barlow_Twins_EEG
A Novel Negative-sample-free Contrastive Self-Supervised Learning for EEG-Based Motor Imagery Classification

>**In-Nea Wang, Cheol-Hui Lee, Hakseung Kim, and Dong-Joo Kim**

## Introduction 🔥
Motor imagery-based brain–computer interface (MI-BCI) systems convert user intentions into computer commands, aiding the communication and rehabilitation of individuals with motor disabilities. Traditional MI classification relies on supervised learning; however, it faces challenges in acquiring large volumes of reliably labeled data and ensuring generalized high performance owing to specific experimental paradigms. To address these issues, this study proposes a contrastive self-supervised learning (SSL) method that does not require negative samples. A MultiResolutionCNN backbone, designed to capture temporal and spatial electroencephalogram (EEG) features, is introduced. By using Barlow Twins loss-based contrastive SSL, features are effectively extracted from EEG signals without labels. MI classification was performed with data from two datasets that were not used for pretraining: an in-domain dataset with two classes and out-domain dataset with four classes. The proposed framework outperformed conventional supervised learning, achieving accuracies of 82.47% and 43.19% on the in-domain and out-domain datasets, respectively. Performance variations were observed in response to variations in the values of key hyperparameters during training. The maximum differences in lambda, learning rate, and batch size were 0.93%, 3.28%, and 3.38%, respectively, demonstrating robust generalization. This study facilitates the development and application of more practical and robust MI-BCI systems.

## Model Architecture
<p align="center"> <img src="https://github.com/dlcjfgmlnasa/Barlow_Twins_EEG/blob/main/figure/figure1.png" alt="image" width="80%" height="auto"> </p>

(A) Illustration of the proposed contrastive SSL using Barlow Twins (B) Architecture of the proposed model. The purple and navy regions represent the backbone and classifier networks of the proposed model, respectively

## Main Result 🥇
#### Comparison with Other Supervised Learning Approaches
<table><thead>
  <tr>
    <th></th>
    <th colspan="3">OpenBMI</th>
    <th colspan="3">SingleArmMI</th>
  </tr></thead>
<tbody>
  <tr>
    <td></td>
    <td> <b> ACC </b> </td>
    <td> <b> MF1 </b> </td>
    <td> <b> κ </b> </td>
    <td> <b> ACC </b> </td>
    <td> <b> MF1 </b> </td>
    <td> <b> κ </b> </td>
  </tr>
  <tr>
    <td>FBCSP</td>
    <td>61.03±4.46</td>
    <td>60.5±14.55</td>
    <td>0.22±0.09</td>
    <td>24.86±1.76</td>
    <td>21.84±3.48</td>
    <td>-0.00±0.02</td>
  </tr>
  <tr>
    <td>ShallowConvNet</td>
    <td>78.95±5.19</td>
    <td>78.93±5.22</td>
    <td>0.58±0.10</td>
    <td>37.36±2.74</td>
    <td>36.23±3.22</td>
    <td>0.16±0.04</td>
  </tr>
  <tr>
    <td>DeepConvNet</td>
    <td>77.39±5.08</td>
    <td>77.37±5.08</td>
    <td>0.55±0.10</td>
    <td>36.39±1.64</td>
    <td>35.90±1.56</td>
    <td>0.15±0.02</td>
  </tr>
  <tr>
    <td>EEGNet</td>
    <td>79.45±4.68</td>
    <td>79.41±4.69</td>
    <td>0.59±0.09</td>
    <td>33.75±3.62</td>
    <td>33.76±3.67</td>
    <td>0.12±0.05</td>
  </tr>
  <tr>
    <td> <b> Proposed Model (linear-prob) </b> </td>
    <td>64.91±5.12</td>
    <td>64.82±5.18</td>
    <td>0.30±0.10</td>
    <td>34.58±4.53</td>
    <td>34.38±4.61</td>
    <td>0.13±0.06</td>
  </tr>
  <tr>
    <td> <b> Proposed Model (fine-tuning) </b> </td>
    <td><b>82.47±4.04</b></td>
    <td><b>82.45±4.05</b></td>
    <td><b>0.65±0.08</b></td>
    <td><b>43.19±3.92</b></td>
    <td><b>42.79±4.13</b></td>
    <td><b>0.24±0.05</b></td>
  </tr>
</tbody></table>

#### UMAP visualization & Confusion Matrix

<p align="center"> <img src="https://github.com/dlcjfgmlnasa/Barlow_Twins_EEG/blob/main/figure/figure2.png" alt="image" width="80%" height="auto"> </p>


## License and Citation 📰
The software is licensed under the Apache License 2.0. Please cite the following paper if you have used this code:

