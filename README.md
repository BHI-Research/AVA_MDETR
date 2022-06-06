# Action Detection using MDETR
Repository dedicated to run Atomic Visual Actions (AVA) into Modulated Detection Transformer (MDETR) for action recognition.

* Paper >> 

* Run the model >> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BHI-Research/AVA_MDETR/blob/main/examples/AVA_MDETR.ipynb)

For our test, we use [AVA Action’s dataset v2.2](https://research.google.com/ava/index.html) which contains 430 videos split into training, validation and test, where each video has 15 minutes annotated in 1 second intervals. Despite of subdivisions it has, we use only videos associated with validation that uses 60 different action's classes. That is because MDETR architecture does not require any training or test set for learning or contrast results.

To process videos, we use MDETR (Modulated Detection for End-to-End Multi-Modal Understanding), which points to be a way to process image detection, conditioned on a raw text query, like a caption or a question. As it was said, MDETR does not require any training stage to start testing. We used it to manage videos, where we disassembled it, taking one image per frame, and processing it.

By default, into Colab file, it analyses only 1 video for not overpassing Google Colab limit. Inside cell called 'Main Sets' principal parameters are configured, like number of videos to analyse and porcentual confidence obtained into the results.

For benchmarking, we use [Activity-Net](https://github.com/activitynet/ActivityNet) as video benchmark for human activity understanding. ActivityNet aims at providing a semantic organization of videos depicting human activities. It points to covering a wide range of complex human activities that are of interest to people in their daily living. ActivityNet provides samples from 203 activity classes with an average of 137 untrimmed videos per class and 1.41 activity instances per video, for a total of 849 video hours. There are three scenarios in which ActivityNet can be used to compare algorithms for human activity understanding: untrimmed video classification, trimmed activity classification and activity detection. You can read more about it in [this paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Heilbron_ActivityNet_A_Large-Scale_2015_CVPR_paper.pdf).

