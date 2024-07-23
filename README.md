# Barlow_Twins_EEG
A Novel Negative-sample-free Contrastive Self-Supervised Learning for EEG-Based Motor Imagery Classification

>**In-Nea Wang, Cheol-Hui Lee, Hakseung Kim, and Dong-Joo Kim**

## Introduction 🔥
Motor imagery-based Brain-computer interfaces (MI-BCI) systems convert user intentions into computer commands, which are particularly useful in providing communication means and aiding rehabilitation for individuals with motor disabilities. Conventionally, MI classification research has focused on supervised learning to extract features from complex brain waves. However, the primary challenge of supervised learning approaches is in acquiring large volumes of reliably labeled high-quality data. Dependence on labeled data is restricted by specific experimental paradigms and protocols, making it challenging to ensure generalized high performance of the supervised learning-based models. To address these challenges, this study proposes a contrastive self-supervised learning (SSL) method for MI classification that does not require negative samples. Additionally, a backbone network called MultiResolutionCNN has been introduced, designed to consider various temporal and spatial scales of raw electroencephalogram (EEG) signals. Utilizing Barlow Twins loss-based contrastive SSL, features are effectively extracted from EEG signals without using negative samples and labels. MI classification was performed on two datasets not used in training: an in-domain dataset with two classes and an out-domain dataset with four classes, achieving accuracies of 81.56% and 42.36%, respectively. The results demonstrated superior performance compared to supervised learning methods. An ablation study compared MultiResolutionCNN with the baseline backbone networks and demonstrated excellent performance, especially when fine-tuning was applied. Additionally, the performance variations were observed with changes in hyperparameters. The maximum differences in accuracy for lambda, learning rate, and batch size were 1.11%, 3.58%, and 3.22%, respectively, indicating robust generalization of performance.

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
    <td>64.87±5.15</td>
    <td>64.78±5.20</td>
    <td>0.30±0.10</td>
    <td>32.92±1.58</td>
    <td>32.69±1.35</td>
    <td>0.11±0.02</td>
  </tr>
  <tr>
    <td> <b> Proposed Model (fine-tuning) </b> </td>
    <td>81.56±4.70</td>
    <td>81.54±4.71</td>
    <td>0.63±0.09</td>
    <td>42.36±2.98<br></td>
    <td>41.96±3.28<br></td>
    <td>0.23±0.04</td>
  </tr>
</tbody></table>

#### Semi-Supervised Learning
#### UMAP visualization & Confusion Matrix


## License and Citation 📰
The software is licensed under the Apache License 2.0. Please cite the following paper if you have used this code:

