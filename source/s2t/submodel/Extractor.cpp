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
#include "Extractor.h"
#include "../../niutensor/tensor/function/GELU.h"
#include "../../niutensor/tensor/core/CHeader.h"
namespace s2t{
    Extractor::Extractor()
    {
        isTraining = false;
        devID = -1;
        inSize = -1;
        hSize = -1;
        nConv = -1;
        kernels= NULL;
        dropoutP = 0.0F;
    }
    Extractor::~Extractor()
    {
        delete[] kernels;
    }
    /* set the training flag */
    void Extractor::SetTrainingFlag(bool myIsTraining)
    {
        isTraining = myIsTraining;
    }

    void Extractor::InitModel(S2TConfig& config)
    {
        SetTrainingFlag(config.training.isTraining);
        devID = config.common.devID;
        nConv = config.s2tmodel.nConv;
        convKernels.assign(config.s2tmodel.convKernel.begin(), config.s2tmodel.convKernel.end());
        convStrides.assign(config.s2tmodel.convStride.begin(), config.s2tmodel.convStride.end());

        inSize = config.s2tmodel.fbank;
        hSize = config.model.encEmbDim;
        kernels = new XTensor[nConv];
        biases = new XTensor[nConv];
        for (int i = 0; i < nConv; i++)
        {
            if (i == 0)
            {
                InitTensor3D(&kernels[i], hSize, inSize, convKernels[i], X_FLOAT, devID);
            }
            else
            {
                InitTensor3D(&kernels[i], hSize, hSize, convKernels[i], X_FLOAT, devID);
            }
            InitTensor1D(&biases[i], hSize, X_FLOAT, devID);
        }
        
    }
    XTensor Extractor::Make(XTensor& input)
    {
        XTensor outFeature;
        outFeature = Conv1DBias(input, kernels[0], biases[0], convStrides[0], 1);
        outFeature = GELU(outFeature);
        for (int i = 1;i<nConv; ++i) 
        {
            outFeature = Conv1DBias(outFeature, kernels[i], biases[i], convStrides[i], 1);
            outFeature = GELU(outFeature);
            }
        return outFeature;
    }
}
