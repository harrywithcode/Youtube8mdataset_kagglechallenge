# Kaggle Competetion-Youtube 8m Video Dataset

## Introduction  <a name="introduction"></a>
We  are  looking  to  take  part  in  one  of  the  popular  competitions  hosted  in  Kaggle:  GoogleCloud  and  YouTube-8M  Video  Understanding  Challenge.  
The  challenge  is  to  develop  videoclassification algorithms which accurately assign video-level labels (multi-labels per video). Since the size of the video database is extremely big, we will first look to solve the scaled downproblem using relatively less number of videos from the entire training set. Just like the well–known Image-Net Challenge which involves  classification of images intomany categories, here, the video labels are generated from a vocabulary of 4716 classes with 3.4 labels per video on average. So, the task is to assign multiple labels(=5) with some confidence (probabilities) to each video.To accomplish this task, we will show the performances of multiple models other than the onesused for benchmarking.

## Dataset <a name="dataset"></a>
Large scale datasets are crucial to research in image and video classification and understanding.They allow researchers to train, validate and test their models and classifiers on a diverse set ofobjects. There exist a number of such datasets with appropriate labels in the image processingcommunity, such as ImageNet[1], which have played a very important role in pushing imagerecognition  and computer vision to near-human  accuracy. In this project we will be using therecent introduced YouTube-8M dataset [2], recently introduced by google, for multi-label videoclassification. The dataset is available at this link for download(https://research.google.com/youtube8m/). There exist some other videosets, such as Sports-1M (for sport videos) and Activity Net (for human activities), but they arelimited to a single category of videos.Some notable features of the YouTube-8M dataset are:

* More than 8 million videos
* Over 500,000 hours of video
* Videos labeled through a vocabulary of 4800 entities generated through the YouTube annotation system
* Each entity contains at least 200 videos with each video viewed at least 1000 times

The main topic of each video is visually recognizable without deep domain expertiseAnother very important feature of the dataset is that it provides frame-level (at one frame persecond) hidden representation one layer prior to the classification layer using a deep convolutionalneural network trained on ImageNet dataset. This amounts to more than 1.9 billion frame levelfeatures. This pre-extraction of the features is very useful in quickly training the models/classifiers
over  the  dataset.  The  extracted  features  are  further  compressed  by  PCA  and  quantization.  Inaddition to frame-level features, video-level fixed length features have also been provided.

## References <a name="references"></a>

[1] Imagenet - http://image-net.org/
[2] Youtube 8m dataset - https://arxiv.org/abs/1609.08675
