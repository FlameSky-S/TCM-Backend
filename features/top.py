from .face import FaceColor, FaceLight, LipColor
from .pulse import PulseCondition, Hemodynamics
from .tongueTop import CoatingColor, TongueColor, TongueCrack, TongueSeg, TongueThickness, TongueWidth, ToothMark
from .tongueBottom import TongueBottomSeg, VeinsColor, VeinsSeg, VeinsIndex

class FeatureExtractor():
    def __init__(self):
        self.modality_map = {
            "face": self.__FaceExtractor,
            "pulse": self.__PulseExtractor,
            "tongue_bottom": self.__TongueBottomExtractor,
            "tongue_top": self.__TongueTopExtractor
        }

        self.face_feat_map = {
            "face_color": FaceColor,
            "face_light": FaceLight,
            "lip_color": LipColor
        }

        self.pulse_feat_map = {

        }

        self.tongue_bottom_map = {

        }

        self.tongue_top_map = {

        }

    def __FaceExtractor(self):
        pass

    def __PulseExtractor(self):
        pass

    def __TongueBottomExtractor(self):
        pass

    def __TongueTopExtractor(self):
        pass

    def __FaceExtractor(self):
        pass

    def do(self, modalities, dataset_dir=None, file_path=None):
        """
        parameters:
            modalities: 模态列表
            file_path: 
        return:
            res: 特征结果pandas.DataFrame -> [sample_id, ...]
        """
        assert dataset_dir or file_path, "at least one of dataset_dir and file_path has to be appointed..."
        self.dataset_dir = dataset_dir
        pass