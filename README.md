# Human Presence Detection via Deep Learning of Passive Radio Frequency Data
Human occupancy detection (HOD) in an enclosed space, such as indoors or inside of a
vehicle, via passive cognitive radio (CR) is a new and challenging research area. Part of the difficulty
arises from the fact that a human subject cannot easily be detected due to spectrum variation. This paper presents a new approach that utilizes software defined radio to passively collect radio frequency data and 
applying deep learning neural network to detect human presence. 
It provides a low cost and environment friendly solution. 
The results of this experiment indicate that human presence 
can be detected by passive RF wireless signals via deep learning 
neural network in a closed space. Robustness is verified by 
testing against different frequency bands, locations and time 
periods.  (https://ieeexplore.ieee.org/document/9438476)
![image](https://user-images.githubusercontent.com/90142624/169698933-6a187a87-0642-4da6-9db2-8f952d3cdf40.png)
![image](https://user-images.githubusercontent.com/90142624/169698917-8c15ec3f-fe0b-43d0-8465-bde6bcd051a3.png)
![image](https://user-images.githubusercontent.com/90142624/169698902-5bbb2d76-6a8c-48fc-8a5f-65a9801169a5.png)

# Human Occupancy Detection via Passive Cognitive Radio
In this paper, we present an advanced HOD system that dynamically reconfigures a CR to collect passive
radio frequency (RF) signals at different places of interest. Principal component analysis (PCA) and
recursive feature elimination with logistic regression (RFE-LR) algorithms are applied to find the
frequency bands sensitive to human occupancy when the baseline spectrum changes with locations.
With the dynamically collected passive RF signals, four machine learning (ML) classifiers are applied
to detect human occupancy, including support vector machine (SVM), k-nearest neighbors (KNN),
decision tree (DT), and linear SVM with stochastic gradient descent (SGD) training. The experimental
results show that the proposed system can accurately detect human subjects—not only in residential
rooms—but also in commercial vehicles, demonstrating that passive CR is a viable technique for
HOD. More specifically, the RFE-LR with SGD achieves the best results with a limited number of
frequency bands. The proposed adaptive spectrum sensing method has not only enabled robust
detection performance in various environments, but also improved the efficiency of the CR system in
terms of speed and power consumption. (https://www.mdpi.com/1424-8220/20/15/4248)

![image](https://user-images.githubusercontent.com/90142624/169699268-ee32e640-a8dc-4f34-b93b-f35d447890fe.png)
![image](https://user-images.githubusercontent.com/90142624/169699300-1deb0ed6-b37d-4c6b-a716-d363c903cc9a.png)
![image](https://user-images.githubusercontent.com/90142624/169699326-d50ba7a7-77f9-4ad0-b22a-facb5884b5ae.png)
![image](https://user-images.githubusercontent.com/90142624/169699364-f6157807-0c2b-40ec-b326-c4b0b885daff.png)
# Synthesis of Passive Human Radio Frequency Signatures via Generative Adversarial Network
a wireless environment can be easily interfered by jamming signals or by 
replaying recorded samples. Hence, the knowledge of the RF 
environment is a critical aspect of a passive RF signals-based 
security monitoring system. Instead of retraining detectors with 
newly collected data, future systems can adapt to a new 
environment by predicting the RF signatures with human 
occupancy given the baseline spectrum of the environment 
measured without human occupancy. Synthesizing RF 
signatures of human occupancy is a challenging research area 
due to the lack of prior knowledge of how a human body alters 
the RF data. A human RF signatures generation system via
conditional generative adversarial networks (GAN) is proposed 
in this paper to synthesize spectrum with human occupancy
using the baseline spectrum at the area of interest. First, the 
trained human RF signatures GAN (HSGAN) model synthesizes 
passive RF signals with human occupancy via the baseline 
spectrum without human occupancy collected in the enclosed 
space. Second, the trained HSGAN model predicts the human 
RF signatures in the enclosed space at a new location using the
HSGAN model trained in other locations. Lastly, the HSGAN
model is quantitatively evaluated via two classifiers including a 
convolutional neural network (CNN) model and a k-nearest 
neighbors (KNN) classifier for the quality of the synthesized 
spectrum. In addition, a 99.5% correlation between synthesize 
human RF signatures and real human RF signatures results
from the HSGAN.(https://ieeexplore.ieee.org/document/9058116)
![image](https://user-images.githubusercontent.com/90142624/169699468-d9156577-e335-4313-8a1c-52270b090c7e.png)
![image](https://user-images.githubusercontent.com/90142624/169699434-26812297-5cfc-486a-963a-d74813bae5ea.png)
![image](https://user-images.githubusercontent.com/90142624/169699499-0fb1ab93-0933-43c2-8e1c-45d29c87b7db.png)

