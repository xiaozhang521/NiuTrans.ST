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

#ifndef __S2TENCODER_H__
#define __S2TENCODER_H__

#include "../nmt/Encoder.h"
#include "S2TConfig.h"
#include "submodel/Extractor.h"
using namespace nmt;
namespace s2t
{
class S2TAttEncoder : public AttEncoder
{
public:
    /*Speech feature extractor*/
    Extractor* extractor;

    /*Postion embedding matrix*/
    XTensor posEmbeddingBase;

    /* constructor */
    S2TAttEncoder();

    /* de-constructor */
    ~S2TAttEncoder();

    /* initialize the model */
    void InitModel(S2TConfig& config);

    /* run encoding for inference with post-norm */
    XTensor RunFastPreNorm(XTensor& input, XTensor* mask);
};
}
#endif