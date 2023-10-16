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

        extractor = new Extractor;
        extractor->InitModel(config);
        
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
        embedder.MakePosEmbedding(posEmbeddingBase, config.model.encEmbDim, config.model.maxSrcLen, config.model.pad, config.common.devID);

        for (int i = 0; i < nlayer; i++) {
            ffns[i].InitModel(config, true);
            selfAtts[i].InitModel(config, true, true);
            attLayerNorms[i].InitModel(config, devID, embDim, config.model.encoderL1Norm);
            fnnLayerNorms[i].InitModel(config, devID, embDim, config.model.encoderL1Norm);
        }
    }

    XTensor S2TAttEncoder::RunFastPreNorm(XTensor& input, XTensor* mask)
    {

        XTensor x = input;
        // FILE* encoderInput = fopen("../tools/data/encoderInput.bin", "rb");
        // x.BinaryRead(encoderInput);

        // conv
        x = Transpose(x, 1, 2);
        x = extractor->Make(x);
        x = Transpose(x, 1, 2);

        // FILE* convOutput = fopen("../tools/data/convOutput.bin", "wb");
        // x.BinaryDump(convOutput);

        XTensor posEmbedding = Unsqueeze(posEmbeddingBase, 0, x.GetDim(0));
        SumMe(x, posEmbedding);

        // FILE* posOutput = fopen("../tools/data/posOutput.bin", "wb");
        // x.BinaryDump(posOutput);

        for (int i = 0; i < nlayer; i++) {

            XTensor xn;
            
            /* layer normalization with pre-norm for self-attn */
            xn = attLayerNorms[i].Run(x);

            /* self attention */
            xn = selfAtts[i].Make(xn, xn, xn, mask, NULL, SELF_ATT);

            /* residual connection */
            SumMe(xn, x);

            /* layer normalization with pre-norm for ffn */
            x = fnnLayerNorms[i].Run(xn);

            /* ffn */
            x = ffns[i].Make(x);

            /* residual connection */
            SumMe(x, xn);

            // break;

        }

        // FILE* blocksOutput = fopen("../tools/data/blocksOutput.bin", "wb");
        // x.BinaryDump(blocksOutput);

        // FILE* lnPostInput = fopen("../tools/data/lnPostInput.bin", "rb");
        // x.BinaryRead(lnPostInput);

        if (finalNorm) {
            return encoderLayerNorm->Run(x); 
        }

        return x;
    }
}
