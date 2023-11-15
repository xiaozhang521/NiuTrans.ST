/* NiuTrans.NMT - an open-source neural machine translation system.
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
 * $Created by: Chi Hu (huchinlp@gmail.com) 2021-11-06
 */

#include <iostream>
#include "./nmt/Config.h"
#include "./nmt/train/Trainer.h"
#include "./nmt/translate/Translator.h"
#include "./s2t/S2TModel.h"
#include "./s2t/generate/Generator.h"
#include "./s2t/S2TVocab.h"
#include "niutensor/tensor/function/GELU.h"

#include "./s2t/WaveLoader.h"
#include "./s2t/FeatureWindow.h"
#include "./s2t/Fbank.h"


using namespace nmt;
using namespace s2t;
using namespace nts;

int main(int argc, const char** argv)
{

    //--------------------------Load Wave--------------------------
    /*
    ifstream inFile("C:\\Code\\VS\\NiuTrans.ST\\test.wav", ios::in | ios::binary);
    if (!inFile) {
        cout << "error no file" << endl;
        return 0;
    }
    class WaveInfo wave;
    class WaveData data;
    data.Read(inFile);
    struct FrameExtractionOptions opt;
    struct FbankOptions fOpts;
    class FbankComputer computer(fOpts);
    class OfflineFeatureTpl<FbankComputer> oft(computer);
    XTensor out;
    oft.ComputeFeatures(data.Data(), data.SampFreq(), 1.0, &out);
    FILE* outputFile = fopen("C:\\Code\\VS\\NiuTrans.ST\\output.txt", "w");
    out.Dump(outputFile);
    fclose(outputFile);
    */
    //--------------------------Load Wave--------------------------
    
    if (argc == 0)
        return 1;
    // load configurations 
    S2TConfig config(argc, argv);
    S2TModel model;
    model.InitModel(config);

    //--------------------------Load Wave--------------------------
    struct FbankOptions fOpts(config);
    class FbankComputer computer(fOpts);
    class OfflineFeatureTpl<FbankComputer> oft(computer);
    //--------------------------Load Wave--------------------------

    Generator generator;
    generator.Init(config, model, oft);
    generator.Generate();
    return 0;
}