# AVA_MDETR
Repository dedicated to run Atomic Visual Actions (AVA) into Modulated Detection Transformer (MDETR) for action recognition.

Paper >> 

To run the model >> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BHI-Research/AVA_MDETR/blob/main/examples/AVA_MDETR.ipynb)

For our test, we use [AVA Action’s dataset v2.2](https://research.google.com/ava/index.html) which contains 430 videos split into training, validation and test, where each video has 15 minutes annotated in 1 second intervals. Despite of subdivisions it has, we use only videos associated with validation. That is because MDETR architecture does not require any training or test set for learning or contrast results.

By default, into Colab file, it analyses only 1 video for not overpassing Google Colab limit. Inside cell called 'Main Sets', you can modify principal parameters like number of videos to analyse and the confidence obtained into the results.

For benchmarking, we use [Activiy-Net](https://github.com/activitynet/ActivityNet) as video benchmark for human activity understanding. 