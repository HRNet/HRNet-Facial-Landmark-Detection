import hrnet_face_landmark.models as models
from hrnet_face_landmark.config import config


__all__ = [
    'get_hrnet',
]


def get_hrnet():
    model = models.get_face_alignment_net(config)
    return model

