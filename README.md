# On Density Map Estimation and Crowd Counting with Convolutional Neural Networks

Crowd counting has become a major focus in computer vision for purposes of crowd control and public safety. The density map enables the user to get more accurate and comprehensive information about the spatial distribution considering the perspectives of the picture, which could be critical for making correct decisions in high-risk environments. The most successful approach involve the use of CNN to reconstruct density map of crowded images. In this project we analyze the well establish CSRNet, proposed by Li et al. [1], challenging the use of dilation in convolutional layers as a possible solution for the task. We also investigated the relationship between the resolution of the training images and the use of dilation. Moreover we propose a new technique to create asymmetric density maps that we improved the MRE score by approximately 13 %. Finally we introduce a promising flux base approach to estimate the variation in the people count.


References:

[1] Yuhong Li, Xiaofan Zhang, and Deming Chen. Csrnet: Dilated convolutional neural networks for understanding the highly congested scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1091–1100, 2018
