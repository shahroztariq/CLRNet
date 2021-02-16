# Overview
__Title:__ *One Detector to Rule Them All: Towards a General Deepfake Attack Detection Framework*

<img src="https://i.ibb.co/8Pf6Chb/CLRNet-pipeline.png" alt="CLRNet-pipeline" border="0" width="800">

# Additional Results

__Updated in-domain attack results including DFDC dataset__

* Note that CLRNet performs the best for DFDC dataset among all the test baselines.

<img src="https://i.ibb.co/HP5dSJF/Table3.png" alt="Table3" border="0" width="600" > 


__Updated out-of-domain attack results (before using our defense strategy)__

* Note that results from Table 5 demonstrates that models trained on DFDC, which is a quite generic and diverse dataset, can still fail to detect out-of-domain attack (see Table 5).
* See Table 6 in our paper, for defense performance against out-of-domain attack.

<img src="https://i.ibb.co/6DSdfWm/Supplementary-DFDC-OOD.png" alt="Supplementary-DFDC-OOD" border="0" width="600">

# Dataset used for Evaluation
* __Facial Reenactment__
  * Face2Face (F2F) [[Dataset]](https://github.com/ondyari/FaceForensics) [[Paper]](https://openaccess.thecvf.com/content_cvpr_2016/html/Thies_Face2Face_Real-Time_Face_CVPR_2016_paper.html)
  * Neural Texture (NT) [[Dataset]](https://github.com/ondyari/FaceForensics) [[Paper]](https://dl.acm.org/doi/abs/10.1145/3306346.3323035)
* __Identity Swap__
  * DeepFake (DF) [[Dataset]](https://github.com/ondyari/FaceForensics) [[GitHub]](https://github.com/deepfakes/faceswap)
  * FaceSwap (FS) [[Dataset]](https://github.com/ondyari/FaceForensics) [[GitHub]](https://github.com/MarekKowalski/FaceSwap/)
  * DeepFake Detection (DFD) [[Dataset]](https://github.com/ondyari/FaceForensics) [[Blog]](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)
  * DeepFake Detection Challenge (DFDC) [[Dataset]](https://dfdc.ai/login) [[Paper]](https://arxiv.org/abs/2006.07397?fbclid=IwAR3_VNnOvjhY8lKe7OIYC4t9jY7RkxrxV0nvpo781o_QdnM25mhjUBj5YUk) [[Kaggle Challenge]](https://www.kaggle.com/c/deepfake-detection-challenge/overview) [[Blog]](https://ai.facebook.com/datasets/dfdc/)
* __Unknown__
  * DeepFake in the Wild (DFW) [[Sample Links]](dataset/DFW_sample_links.md) [[Source1]](https://onedualityfakes.com/Forum-Share-your-fakes-SFW-only) [[Source2]](https://forum.faceswap.dev/viewforum.php?f=8&sid=b4c58dff0d5889ec37acfc28c5a2de1c) [[Source3]](https://www.reddit.com/r/SFWdeepfakes/)

# Models used for Evaluation
* __Single Frame-based__
  * ShallowNet [[Paper]](https://dl.acm.org/doi/abs/10.1145/3267357.3267367) [[GitHub]](https://github.com/shahroztariq/ShallowNet)
  * Xception [[Paper]](https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html) [[GitHub]](https://github.com/fchollet/deep-learning-models/blob/master/xception.py)
  * MesoNet [[Paper]](https://ieeexplore.ieee.org/abstract/document/8630761) [[GitHub]](https://github.com/DariusAf/MesoNet)
* __Multiple Frame-based__
  * CNN+LSTM [[Paper]](https://ieeexplore.ieee.org/abstract/document/8639163)
  * DBiRNN [[Paper]](https://openaccess.thecvf.com/content_CVPRW_2019/html/Media_Forensics/Sabir_Recurrent_Convolutional_Strategies_for_Face_Manipulation_Detection_in_Videos_CVPRW_2019_paper.html)
  * CLRNet (Ours)
