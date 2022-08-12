# hrnet_face_landmark

## Summary

This is wrapper for [HRNet/HRNet-Facial-Landmark-Detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection)

## Features

- [x] face landmark detection with hrnet as library
- [ ] face tracking
- [ ] automation for downloading and cropping the video from url

## Installation

### With source

```
git clone git@github.com:kingsj0405/hrnet_face_landmark.git
pip install -e .
```

### TODO: With pip
```
pip install hrnet_face_landmark
```

## TODO: Usage

```
from hrnet_face_landmark import HRNet

data = torch.randn(1, 3, 256, 256)
feat = HRNet.get_facial_feature(data)
pred = HRNet.get_facial_landmark(data)

image_path = "./data/image.png"
cropped_image = HRNet.crop_image(image_path)

video_path = "./data/video.mp4"
cropped_video_path = "./data/video_cropped.mp4"
cropped_frame_path = "./data/video_cropped/"
HRNet.crop_video(video_path, cropped_video_path, cropped_frame_path)
```

## Citation

```
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}
```

## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}
````

## Reference
[1] Deep High-Resolution Representation Learning for Visual Recognition. Jingdong Wang, Ke Sun, Tianheng Cheng, 
    Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, Bin Xiao. Accepted by TPAMI.  [download](https://arxiv.org/pdf/1908.07919.pdf)

