import cv2
import os
import Classify.DataPreparation as dp
import Classify.ClassifyScreen as cs

class ScreenClassifier:
    def __init__(self, model_path = "C:\\pycharm\\files\\kea-final\\Classify\\Saved_Models\\rico_screen_classifier_extenddata_acc81.pth"):
        self.model = cs.initClassifierModel(model_path)

    def get_pred(self, current_screen_pth):
        if not current_screen_pth.lower().endswith('.jpg'):
            current_screen_image = cv2.imread(current_screen_pth)
            dir_path, file_name = os.path.split(current_screen_pth)
            temp_jpg_pth = os.path.join(dir_path, file_name[:-4] + '.jpg')
            cv2.imwrite(temp_jpg_pth, current_screen_image)
            # print("saved"+str(temp_jpg_pth))
        else:
            temp_jpg_pth = current_screen_pth

        ocr, sil = dp.createOCR_text_layout(temp_jpg_pth)
        preds = cs.classifyScreen(temp_jpg_pth, sil, ocr, self.model)
        return preds

