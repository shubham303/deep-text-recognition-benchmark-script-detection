"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from modules.Transformer import Seq2SeqTransformer
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):

    def __init__(self, opt,character):
        super(Model, self).__init__()
        self.opt = opt
        #self.character is passed to prediction model to mask certain characters during prediction of regex based text
        self.character = character
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM' and opt.Prediction != "transformer":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
       
        if opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        elif opt.Prediction == "transformer":
            #  ref ;https://pytorch.org/tutorials/beginner/translation_transformer.html
            self.Prediction = Seq2SeqTransformer(opt.encoder_count, opt.decoder_count, self.SequenceModeling_output,
                                                 opt.attention_heads,
                                                 opt.num_class,
                                                 opt.hidden_size)
            for p in self.Prediction.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            raise Exception('Prediction is neither Attn or transformer')

    # regex is used if we expect predicted text to follow certain pattern. ex: regex for PAN number is "[A-Z]{5}[
    # 0-9]{4}[A-Z]{1}" so here for first five positions, predicted probablities of numbers and special characters
    # are set to -infinity.
    def forward(self, input, text, is_train=True, regex=None, src_mask= None, tgt_mask=None, tgt_padding_mask=None):

        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

      
        """ Sequence modeling stage """
        #BiLSTM encoder is not used in transformer.
        if self.stages['Seq'] == 'BiLSTM'  and self.stages['Pred'] != "transformer":
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM



        """ Prediction stage """
        if self.stages['Pred']== "Attn":
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train,
                                        self.opt.batch_max_length,regex,  self.character)
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, regex ,
                                         self.character, tgt_mask, tgt_padding_mask)
        return prediction
