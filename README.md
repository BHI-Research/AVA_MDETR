# Action Detection using MDETR
This repository contains code and link to run Atomic Visual Actions (AVA) into Modulated Detection Transformer (MDETR) for action recognition.

* Paper >> 

* Run the model >> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BHI-Research/AVA_MDETR/blob/main/examples/AVA_MDETR.ipynb)

## Introduction
For our test, we use [AVA Action’s dataset v2.2](https://research.google.com/ava/index.html) which contains 430 videos split into training, validation and test, where each video has 15 minutes annotated in 1 second intervals. Despite of subdivisions it has, we use only videos associated with validation that uses 60 different action's classes. That is because MDETR architecture does not require any training or test set for learning or contrast results.

When code is run, is starts downloading some files, saving questions and actions list. Then the model in instanced, and each video is streamed from AVA Action's dataset repository. Frames are processed, and results are saved in a CSV file, to later compare it.

To process videos, we use [MDETR](https://github.com/ashkamath/mdetr) (Modulated Detection for End-to-End Multi-Modal Understanding), which points to be a way to process image detection, conditioned on a raw text query, like a caption or a question. As it was said, MDETR does not require any training stage to start testing. We used it to manage videos, where we disassembled it, taking one image per frame, and processing it. By default, MDETR provides several engines, trained with differents parameters, and three "downstream tasks", which are methods used by MDETR to process information. After a group of tests, we decided that "Question Answering" task was the most functional for our problem. You can read more about MDETR in [this paper](https://arxiv.org/pdf/2104.12763v2.pdf).

By default, into Colab file, it analyses only 1 video for not overpassing Google Colab limit. Inside cell called 'Main Sets' principal parameters are configured, like number of videos to analyse and porcentual confidence obtained into the results.

For benchmarking, we use [Activity-Net](https://github.com/activitynet/ActivityNet) as video benchmark for human activity understanding. ActivityNet aims at providing a semantic organization of videos depicting human activities. It points to covering a wide range of complex human activities that are of interest to people in their daily living. ActivityNet provides samples from 203 activity classes with an average of 137 untrimmed videos per class and 1.41 activity instances per video, for a total of 849 video hours. There are three scenarios in which ActivityNet can be used to compare algorithms for human activity understanding: untrimmed video classification, trimmed activity classification and activity detection. You can read more about it in [this paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Heilbron_ActivityNet_A_Large-Scale_2015_CVPR_paper.pdf).

## Examples
Here can be seen some images processed with MDETR, and plotting to see the rectangle around object detected.
Also, you can the confidence, which is a metric that MDETR provides, to know how sure it is about the result get.
In our code, many detections are discarded because confidence is not high enough. You can change that modifying "CONFIDENCE" constante in the code.

### Success results

| | Images | Images  ||
| :----------:  | :----------:  | :----------:      | :----------:  |
| ![is someone sit](images/is_someone_sit.png)      | ![is someone stand](images/is_someone_stand.png)       |
| ![is someone sleeping](images/is_someone_sleeping.png)   | ![who is giving a handshake](images/who_is_giving_a_handshake.png)        |
| ![who is standing](images/who_is_standing.png)   | ![who is giving a handshake](images/who_is_giving_a_handshake.png)        |

### Failure results
![is someone dancing](images/is_someone_stand.png)

As it can be seem, in some cases MDETR recognized succesfully actions, but in others, it failed. That was expected, because model was not trained
specifically for this dataset, images in AVA are difficult to process and MDETR was not thought to work with actions. All this factors causes that
model fails to recognized some situations in images.

## Results
After running validation dataset of AVA, we get some results. There were actions easily recognized, and others that not.

<center>

| | Pascal Boxes | Category Results  ||
| :----------:  | :----------:  | :----------:      | :----------:  |
| Category      | AP@0.5IOU     | Category          | AP@0.5IOU     |
| Sleep         | 0.0019        | answer phone      | 0.0           |
| Sit           | 0.0016        | kiss (a person)   | 0.0           |   
| Stand Up      | 0.0011        | throw             | 0.0           |
| Handshake     | 0.0005        | touch (an object) | 0.0           |
| Dance         | 0.0005        | touch (an object) | 0.0           |

</center>
