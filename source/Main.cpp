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


using namespace nmt;
using namespace s2t;
using namespace nts;

int main(int argc, const char** argv)
{

    //--------------------------Load Wave--------------------------
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
    return 0;
    //--------------------------Load Wave--------------------------
    //std::ios_base::sync_with_stdio(false);
    //std::cin.tie(NULL);

    if (argc == 0)
        return 1;
    /* load configurations */
    /*S2TConfig config(argc, argv);
    S2TModel model;
    model.InitModel(config);
    config.showConfig();

    model.TestDumpParams(&model.encoder->selfAtts[1].weightQ);

    cout << "Tgt Vocab File: " << config.common.tgtVocabFN << endl;
    S2TVocab vocab;
    vocab.Load(config.common.tgtVocabFN);
    // vocab.ShowVocab();
    vocab.Test();


    Generator generator;
    generator.Init(config, model);
    generator.TestTranslate();*/

    float a[20] = { -7.775555e-01, -7.483220e-01,-7.483220e-01,-7.483220e-01,-7.483220e-01,-7.483220e-01,-7.483220e-01,-7.483220e-01,-7.483220e-01,-7.486424e-01,-7.498121e-01,-7.468143e-01,-7.395701e-01,-8.399765e-01,-1.032664e+00,-1.253703e+00,-1.278807e+00,-1.145470e+00,-9.983076e-01,-8.426485e-01 };
    XTensor input;
    InitTensor2D(&input, 1, 20, X_FLOAT, 0);
    input.SetData(a, input.unitNum);
    input.Dump();
    XTensor output;
    output = GELU(input);
    output.Dump();
    //generator.generate(); 
    // 
    /*****************************Old entrance******************************/
    //srand(config.common.seed);

    ///* training */
    //if (strcmp(config.training.trainFN, "") != 0) {

    //    NMTModel model;
    //    model.InitModel(config);

    //    Trainer trainer;
    //    trainer.Init(config, model);
    //    trainer.Run();
    //}

    ///* translation */
    //else if (strcmp(config.translation.inputFN, "") != 0) {

    //    /* disable gradient flow */
    //    DISABLE_GRAD;

    //    NMTModel model;
    //    model.InitModel(config);

    //    Translator translator;
    //    translator.Init(config, model);
    //    translator.Translate();
    //}
    //else {
    //    fprintf(stderr, "Thanks for using NiuTrans.NMT! This is an effcient\n");
    //    fprintf(stderr, "neural machine translation system. \n\n");
    //    fprintf(stderr, "   Run this program with \"-train\" for training!\n");
    //    fprintf(stderr, "Or run this program with \"-input\" for translation!\n");
    //}


}