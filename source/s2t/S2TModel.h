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

#ifndef __MODEL_S2T__
#define __MODEL_S2T__

#include "S2TConfig.h"
#include "../nmt/Decoder.h"
#include "S2TEncoder.h"
#include "../nmt/submodel/Output.h"
#include "../niutensor/train/XModel.h"
using namespace nts;
  /* the s2t namespace */
namespace s2t
{

    /* an nmt model that keeps parameters of the encoder,
       the decoder and the output layer (softmax). */
    class S2TModel : public XModel
    {
    public:
        /* device id */
        int devID;

        ///* configurations */
        S2TConfig* config;

        /* the encoder */
        S2TAttEncoder* encoder;

        /* the decoder */
        AttDecoder* decoder;

        /* output layer */
        OutputLayer* outputLayer;

    public:
        /* constructor */
        S2TModel();

        /* de-constructor */
        ~S2TModel();

    //    /* get configurations */
        vector<int*> GetIntConfigs();
        vector<bool*> GetBoolConfigs();
        vector<float*> GetFloatConfigs();

        /* initialize the model */
        void InitModel(S2TConfig& config);

    //    /* print model configurations */
    //    void ShowModelConfig();

    //    /* make the encoding network */
    //    XTensor MakeEncoder(XTensor& input, XTensor* mask);

    //    /* make the encoding network */
    //    XTensor MakeDecoder(XTensor& inputEnc, XTensor& inputDec, XTensor* mask,
    //        XTensor& MaskEncDec);

    //    /* make the network for language modeling (with the output softmax layer) */
    //    XTensor MakeLM(XTensor& input, XTensor& padding);

    //    /* make the network for machine translation (with the output softmax layer) */
    //    XTensor MakeMT(XTensor& inputEnc, XTensor& inputDec,
    //        XTensor& paddingEnc, XTensor& paddingDec);

    //    /* make the mask for training MT models */
    //    void MakeMTMask(XTensor& inputEnc, XTensor& inputDec,
    //        XTensor& paddingEnc, XTensor& paddingDec,
    //        XTensor& maskEnc, XTensor& maskDec, XTensor& maskEncDec);

    //    /* make the mask of the encoder */
    //    void MakeMTMaskEnc(XTensor& paddingEnc, XTensor& maskEnc);

    //    /* make the mask of the decoder */
    //    void MakeMTMaskDec(XTensor& paddingEnc, XTensor& paddingDec,
    //        XTensor& maskDec, XTensor& maskEncDec);

    //    /* make the mask of the decoder for inference */
    //    XTensor MakeMTMaskDecInference(XTensor& paddingEnc);

        /* get parameter matrices */
        void GetParams(TensorList& list);

    //    /* dump the model to a file */
    //    void DumpToFile(const char* fn);

    //    /* read the parameters */
    //    void LoadFromFile(FILE* file);

    //    /* get the number of parameters */
    //    uint64_t GetParamNum();

    //    /* set the training flag */
    //    void SetTrainingFlag(bool isTraining);

    //public:

    //    /* clone the model (overloaded method of XModel) */
    //    XModel* Clone(int devID);

    //    /* run the neural network (overloaded method of XModel) */
    //    bool RunSimple(XList* inputs, XList* outputs, XList* golds, XList* losses);
    };

} /* end of the s2t namespace */

#endif /* __MODEL_S2T__ */
