The training data consists of a few hundred videos categorized into one of the five classes. Each video (typically 2-3 seconds long) is divided into a sequence of 30 frames(images). Two types of architectures have been used for above:
1. CNN + RNN : CNN extracts the feature vectors from the image, we then pass the sequence of these feature vectors through an RNN. We use transfer learning for the 2D CNN.
2. Conv3D: the input to a 3D conv is a video. The video becomes a 4-D tensor of shape 100x100x3x30 which can be written as (100x100x30)x3 where 3 is the number of channels. This cubic filter will now '3D-convolve' on each of the three channels of the (100x100x30) tensor.

The data ingestion pipeline is setup using a custom generator.
