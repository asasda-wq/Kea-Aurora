import torch
import torch.nn as nn
from Classify.Saved_Models.clip import clip
from transformers import BertTokenizer, BertModel


class RICOScreenClassifer(nn.Module):
    def __init__(self, classnum, device):
        super().__init__()
        self.device = device
        self.image_encoder,_ = clip.load('ViT-B/32', device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.head = nn.Sequential(
            nn.Linear(1280, 2560),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2560, 5120),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(5120, classnum)
        )

    def forward(self, raw_images, layout_images, texts):
        raw_features = self.image_encoder.encode_image(raw_images).to(self.device)
        layout_features = self.image_encoder.encode_image(layout_images).to(self.device)
        texts_inputs = self.tokenizer(texts,padding='max_length',truncation=True,max_length=256,add_special_tokens=True,return_tensors="pt")
        image_features = 0.5*raw_features + 0.5*layout_features
        input_ids = texts_inputs['input_ids'].to(self.device)
        token_type_ids = texts_inputs['token_type_ids'].to(self.device)
        attention_mask = texts_inputs['attention_mask'].to(self.device)
        text_features = self.bert(input_ids=input_ids,token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        text_features = torch.mean(text_features, dim=1).to(self.device)
        features = torch.cat([text_features, image_features], dim=1).to(self.device)
        output = self.head(features)
        return output
