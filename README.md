# ExplicitOrientationEstimator
PyTorch implementation of ["Explicit Learning of Feature Orientation Estimation"](https://ieeexplore.ieee.org/document/8803644). Ji Dai, Junkang Zhang, and Truong Nguyen. *Proceedings of the IEEE conference on image processing. 2019.*

# Requirement
- PyTorch
- OpenCV
- Matplotlib

# Pretrained model weights
- Please download the weight here: https://drive.google.com/file/d/17U2PyKYXU6UEmmqtyvY2jpOhx0AbT8oL/view?usp=sharing and place it under the root dir.
- This weights is trained with on HPatch dataset for EdgeFociB detector. **Please retrain the model when using other detectors for best performance**. Please follow the training pipeline discussed in the paper.

# Run
- `python test.py`
- This code sample keypoints defined in 'keypoints.txt' on 'image.png'. The code plot a 2 x 2 figure. The top row showes two sampled patches from the sample keypoint with a randomly assigned orientation difference. The subtitles are estimated orientations from the network. The second row showes the rectified patches with estimated orientations, they are supposed to be identical.
