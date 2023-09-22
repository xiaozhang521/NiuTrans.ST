/* NiuTrans.S2T - an open-source speech-to-text system.
 * Copyright (C) 2020 NiuTrans Research. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/*
* $Created by: Yuhao Zhang (yoohao.zhang@gmail.com) 2023-09-19
*/
#include "S2TModel.h"
#include <iostream>
namespace s2t
{
    S2TModel::S2TModel()
    {
        devID = -1;
        config = NULL;
        encoder = new S2TAttEncoder();
        decoder = new AttDecoder();
        outputLayer = new OutputLayer();
    }
    S2TModel::~S2TModel()
    {

    }

    /* return a list to keep the configurations (interger) */
    vector<int*> S2TModel::GetIntConfigs()
    {
        /* 19 integers */
        vector<int*> intConfig = {
            &(config->model.fbank),
            &(config->model.encEmbDim),
            &(config->model.encLayerNum),
            &(config->model.encSelfAttHeadNum),
            &(config->model.encFFNHiddenDim),
            &(config->model.decEmbDim),
            &(config->model.decLayerNum),
            &(config->model.decSelfAttHeadNum),
            &(config->model.encDecAttHeadNum),
            &(config->model.decFFNHiddenDim),
            &(config->model.maxRelativeLength),
            &(config->model.maxSrcLen),
            &(config->model.maxTgtLen),
            &(config->model.sos),
            &(config->model.eos),
            &(config->model.pad),
            &(config->model.unk),
            &(config->model.srcVocabSize),
            &(config->model.tgtVocabSize),
        };

        return intConfig;
    }

    /* return a list to keep the configurations (bool) */
    vector<bool*> S2TModel::GetBoolConfigs()
    {
        /* 12 bool */
        vector<bool*> boolConfig = {
            &(config->model.encoderL1Norm),
            &(config->model.decoderL1Norm),
            &(config->model.useBigAtt),
            &(config->model.decoderOnly),
            &(config->model.encFinalNorm),
            &(config->model.decFinalNorm),
            &(config->model.encPreLN),
            &(config->model.decPreLN),
            &(config->model.useEncHistory),
            &(config->model.useDecHistory),
            &(config->model.shareEncDecEmb),
            &(config->model.shareDecInputOutputEmb),
        };

        return boolConfig;
    }

    /* return a list to keep the configurations (float) */
    vector<float*> S2TModel::GetFloatConfigs()
    {
        /* 3 float */
        vector<float*> floatConfig = {
            &(config->model.dropout),
            &(config->model.ffnDropout),
            &(config->model.attDropout),
        };

        return floatConfig;
    }

    /*
    initialize the model
    >> myConfig - configuration of the model
    */
    void S2TModel::InitModel(S2TConfig& myConfig)
    {
        std::cout << "----- S2TModel Init -----" << std::endl;
        config = &myConfig;
        devID = config->common.devID;

        /* configurations for the model */
        vector<int*> intConfig = GetIntConfigs();
        vector<bool*> boolConfig = GetBoolConfigs();
        vector<float*> floatConfig = GetFloatConfigs();

        FILE* modelFile = NULL;
        modelFile = fopen(config->common.modelFN, "rb");
        cout << "+ modelFile: " << config->common.modelFN << "\t" << (modelFile == NULL) << endl;

        /* read model configurations */
        if (modelFile) {
        
            // CheckNTErrors(modelFile, "Failed to open the model file");

            LOG("loading configurations from the model file...");

            /* 12 booleans */
            for (auto c : boolConfig) {
                fread(c, sizeof(bool), 1, modelFile);
            }
            int maxSrcLen = config->model.maxSrcLen;
            /* 19 intergers */
            for (auto c : intConfig) {
                fread(c, sizeof(int), 1, modelFile);
            }
            /* reset the maximum source sentence length */
            config->model.maxSrcLen = MIN(maxSrcLen, config->model.maxSrcLen);
            /* 3 float */
            for (auto c : floatConfig) {
                fread(c, sizeof(float), 1, modelFile);
            }
        }

        if (config->training.isTraining) {
            
            /* currently we do not support training */

        }

        std::cout << "--- S2TModel Init End ---" << std::endl;

        if (config->training.isTraining)
        {
            ShowNTErrors("TODO!!");
        }
        encoder->InitModel(*config);
        //decoder->InitModel(*config);
        //outputLayer->InitModel(*config);
    }

}