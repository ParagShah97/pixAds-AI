import torch
import numpy as np
import advertisments as ads
import torch.nn as nn
import model_structure as ms
from collections import Counter
from torchvision import transforms
from PIL import Image



class LabelPred():
    def __init__(self):
        self.DenseModel = ms.Densenet(in_channel=3, classes=10)
        self.resnetModel = ms.ResNet18(num_classes=10)
        self.vgg13Model = ms.VGG13(num_classes=10)
        # self.alexnetModel = ms.AlaxNet_custom(in_channels=3, out_classes=10)
        self.DenseModel.load_state_dict(torch.load('model_weights/densenet_final_project_75.pth',  map_location=torch.device('cpu')))
        self.resnetModel.load_state_dict(torch.load('model_weights/rnet_model_accuracy_82.29809406279995.pth', map_location=torch.device('cpu')))
        self.vgg13Model.load_state_dict(torch.load('model_weights/vgg_final_project_70.pth', map_location=torch.device('cpu')))
    
    # label_pred_internal: Internal Method for 
    def label_pred_internal(self, model, image):
        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        predicted_class = ads.class_names[predicted.item()]
        # print(f"The predicted class is: {predicted_class}")
        return predicted_class

    # predict: Method takes the images and return the most reliable label.
    def predict(self, input_image):
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = Image.open(input_image).convert('RGB')
        image = self.transform(image)
        if image.dim() == 3:
            image = image.unsqueeze(0) 
        elif image.dim() == 5:
            image = image.squeeze(0)
        model_inst = [self.resnetModel, self.vgg13Model, self.DenseModel]
        return_labels = []
        for i in range(3):
            return_labels.append(self.label_pred_internal(model_inst[i], image))
        word_counts = Counter(return_labels)
        print('Word count ()', word_counts, return_labels)
        most_frequent_word = word_counts.most_common(1)[0][0]
        return most_frequent_word
    


