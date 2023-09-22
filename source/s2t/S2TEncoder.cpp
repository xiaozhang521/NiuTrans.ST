/* NiuTrans.S2T - an open-source speech to text system.
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
  * $Created by: yuhao zhang(yoohao.zhang@gmail.com) 2023-09-22
  */

#include "S2TEncoder.h"
using namespace nmt;
namespace s2t
{

    S2TAttEncoder::S2TAttEncoder()
    {
        
    }

    
    S2TAttEncoder::~S2TAttEncoder()
    {

    }

    void S2TAttEncoder::InitModel(S2TConfig& config)
    {
        SetTrainingFlag(config.training.isTraining);
        devID = config.common.devID;
        preLN = config.model.encPreLN;
        dropoutP = config.model.dropout;
        embDim = config.model.encEmbDim;
        nlayer = config.model.encLayerNum;
        vSize = config.model.srcVocabSize;
        finalNorm = config.model.encFinalNorm;
        useHistory = config.model.useEncHistory;

        extractor.InitModel(config);
        
        //CheckNTErrors(vSize > 1, "Set vocabulary size by \"-vsize\"");
        CheckNTErrors(nlayer >= 1, "We have one encoding layer at least!");

        ffns = new FFN[nlayer];
        selfAtts = new Attention[nlayer];
        attLayerNorms = new LayerNorm[nlayer];
        fnnLayerNorms = new LayerNorm[nlayer];


        
        if (useHistory) {
            history = new LayerHistory;
            history->InitModel(config, true);
        }

        if (finalNorm) {
            encoderLayerNorm = new LayerNorm;
            encoderLayerNorm->InitModel(config, devID, embDim, config.model.encoderL1Norm);
        }

        /* initialize the stacked layers */
        //embedder.InitModel(config);

        for (int i = 0; i < nlayer; i++) {
            ffns[i].InitModel(config, true);
            selfAtts[i].InitModel(config, true, true);
            attLayerNorms[i].InitModel(config, devID, embDim, config.model.encoderL1Norm);
            fnnLayerNorms[i].InitModel(config, devID, embDim, config.model.encoderL1Norm);
        }
    }
}
