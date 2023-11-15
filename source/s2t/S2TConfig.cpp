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
  *
  */

#include <iostream>
#include <fstream>
#include "S2TConfig.h"

using namespace std;
using namespace nts;

/* the s2t namespace */
namespace s2t
{   
    /*
    load configurations from the command
    >> argc - number of arguments
    >> argv - the list of arguments
    */
    S2TConfig::S2TConfig(int argc, const char** argv) : NMTConfig(argc, argv)
    {   
        cout << "----- S2TConfig Constructor -----" << endl;
        cout << "+ num of params: " << argc << " :: ";
        for (int i=0; i < argc; i++) {
            cout << " " << argv[i];
        }
        cout << endl;

        char** args = new char* [MAX_PARAM_NUM];
        for (int i = 0; i < argc; i++) {
            args[i] = new char[strlen(argv[i]) + 1];
            strcpy(args[i], argv[i]);
        }
        for (int i = argc; i < MAX_PARAM_NUM; i++) {
            args[i] = NULL;
        }

        char* configFN = new char[1024];
        LoadParamString(argc, args, "config", configFN, "");

        cout << "+ configFN: " << configFN << endl;

        int argsNum = argc;

        /* override the configuration according to the file content */
        if (strcmp(configFN, "") != 0)
            argsNum = LoadFromFile(configFN, args);


        // showConfig();
        /* parse configuration in args */
        model.Load(argsNum, (const char **)args);
        s2tmodel.Load(argsNum, (const char**)args);
        common.Load(argsNum, (const char **)args);
        training.Load(argsNum, (const char **)args);
        inference.Load(argsNum, (const char **)args);
        whisperdec.Load(argsNum, (const char**)args);
        extractor.Load(argsNum, (const char**)args);
        // translation = (TranslationConfig)inference;

        // showConfig();

        for (int i = 0; i < MAX(argc, argsNum); i++)
            delete[] args[i];
        delete[] args;
        delete[] configFN;

        cout << "--- S2TConfig Constructor End ---" << endl;
    }

    /*
    load configurations from a file
    >> configFN - path to the configuration file
    >> args - the list to store the configurations
    << argsNum - the number of arguments
    format: one option per line, separated by a blank or a tab
    */
    int S2TConfig::LoadFromFile(const char* configFN, char** args) 
    {
        ifstream f(configFN, ios::in);
        CheckNTErrors(f.is_open(), "Failed to open the config file");

        int argsNum = 0;

        /* parse arguments from the file */
        string key, value;
        while (f >> key >> value && argsNum < (MAX_PARAM_NUM - 1)) {
            cout << "\t- key: "  << key << " value: " << value <<endl;
            if (args[argsNum] != NULL) {
                delete[] args[argsNum];
            }
            if (args[argsNum + 1] != NULL) {
                delete[] args[argsNum + 1];
            }
            args[argsNum] = new char[1024];
            args[argsNum + 1] = new char[1024];
            strcpy(args[argsNum++], key.c_str());
            strcpy(args[argsNum++], value.c_str());
        }

        cout << "+ argsNum: "  << argsNum <<endl;

        /* record the number of arguments */
        return argsNum;
    }

    void S2TConfig::showConfig()
    {
        cout << "+ S2T Model Config" << endl;
        /* 19 integers */
        cout << "    - Integer" << endl;
        cout << "\t| " << "srcVocabSize = " << model.srcVocabSize << endl;
        cout << "\t| " << "tgtVocabSize = " << model.tgtVocabSize << endl;
        cout << "\t| " << "sos = " << model.sos << endl;
        cout << "\t| " << "eos = " << model.eos << endl;
        cout << "\t| " << "pad = " << model.pad << endl;
        cout << "\t| " << "unk = " << model.unk << endl;
        cout << "\t| " << "maxSrcLen = " << model.maxSrcLen << endl;
        cout << "\t| " << "maxTgtLen = " << model.maxTgtLen << endl;
        cout << "\t| " << "maxRelativeLength = " << model.maxRelativeLength << endl;
        cout << "\t| " << "fbank = " << s2tmodel.fbank << endl;
        cout << "\t| " << "encEmbDim = " << model.encEmbDim << endl;
        cout << "\t| " << "encLayerNum = " << model.encLayerNum << endl;
        cout << "\t| " << "encSelfAttHeadNum = " << model.encSelfAttHeadNum << endl;
        cout << "\t| " << "encFFNHiddenDim = " << model.encFFNHiddenDim << endl;
        cout << "\t| " << "decEmbDim = " << model.decEmbDim << endl;
        cout << "\t| " << "decLayerNum = " << model.decLayerNum << endl;
        cout << "\t| " << "decSelfAttHeadNum = " << model.decSelfAttHeadNum << endl;
        cout << "\t| " << "encDecAttHeadNum = " << model.encDecAttHeadNum << endl;
        cout << "\t| " << "decFFNHiddenDim = " << model.decFFNHiddenDim << endl;
        cout << "\t| " << "fnnActFunType = " << model.fnnActFunType << endl;



        /* 12 bool */
        cout << "    - Bool" << endl;
        cout << "\t| " << "encoderL1Norm = " << model.encoderL1Norm << endl;
        cout << "\t| " << "decoderL1Norm = " << model.decoderL1Norm << endl;
        cout << "\t| " << "useBigAtt = " << model.useBigAtt << endl;
        cout << "\t| " << "decoderOnly = " << model.decoderOnly << endl;
        cout << "\t| " << "encFinalNorm = " << model.encFinalNorm << endl;
        cout << "\t| " << "decFinalNorm = " << model.decFinalNorm << endl;
        cout << "\t| " << "encPreLN = " << model.encPreLN << endl;
        cout << "\t| " << "decPreLN = " << model.decPreLN << endl;
        cout << "\t| " << "useEncHistory = " << model.useEncHistory << endl;
        cout << "\t| " << "useDecHistory = " << model.useDecHistory << endl;
        cout << "\t| " << "shareEncDecEmb = " << model.shareEncDecEmb << endl;
        cout << "\t| " << "shareDecInputOutputEmb = " << model.shareDecInputOutputEmb << endl;

        /* 3 float */
        cout << "    - Float" << endl;
        cout << "\t| " << "dropout = " << model.dropout << endl;
        cout << "\t| " << "ffnDropout = " << model.ffnDropout << endl;
        cout << "\t| " << "attDropout = " << model.attDropout << endl;
    }

    /* load s2t model configuration from the command */
    void S2TModelConfig::Load(int argsNum, const char** args)
    {
        Create(argsNum, args);

        LoadInt("fbank", &fbank, 80);

    }

    /* load whisper configuration from the command */
    void WhisperDecConig::Load(int argsNum, const char** args) {
        Create(argsNum, args);
        LoadString("task", task, "transcribe");
        LoadString("lang", language.language, "en");
        LoadFloat("temperature", &temperature, 0.0);
        LoadFloat("nospeechthreshold", &noSpeechThreshold, 0.6);
        LoadFloat("logprobthreshold", &logProbThreshold, -1.0);
        LoadFloat("compratiothreshold", &compRatioThreshold, 2.4);
        LoadBool("notimeStamps", &withoutTimeStamps, false);
        InitLanguageToken();
    }

    void WhisperDecConig::InitLanguageToken() 
    {
        if (strcmp(language.language, "en") == 0) {
            std::cout << "Language:  English" << std::endl;
            language.languageToken = 50259;
        }
        else if (strcmp(language.language, "zh") == 0) {
            std::cout << "Language:  Chinese" << std::endl;
            language.languageToken = 50260;
        }
        else {
            language.languageToken = 50259;
            std::cout << "Unknown Language: " << language.language << ", Decode in English" << std::endl;
        }
    }

    void ExtractionConfig::Load(int argsNum, const char** args) {
        Create(argsNum, args);

        LoadBool("useEnergy", &useEnergy, FALSE);
        LoadFloat("energyFloor", &energyFloor, 0.0);
        LoadBool("rawEnergy", &rawEnergy, TRUE);
        LoadBool("htkCompat", &htkCompat, FALSE);
        LoadBool("useLogFbank", &useLogFbank, TRUE);
        LoadBool("usePower", &usePower, TRUE);
        LoadBool("oneside", &oneSide, FALSE);
        LoadString("inputAudio", inputAudio, "../test.wav");

        LoadFloat("sampFreq", &sampFreq, 16000.0);
        LoadFloat("frameShiftMs", &frameShiftMs, 10.0);
        LoadFloat("frameLengthMs", &frameLengthMs, 25.0);
        LoadFloat("chunkLengthMs", &chunkLengthMs, 30000.0);
        LoadFloat("dither", &dither, 0.0);
        LoadFloat("preemphCoeff", &preemphCoeff, 0.0);
        LoadBool("removeDcOffset", &removeDcOffset, FALSE);
        LoadString("windowType", windowType, "hanning_periodic");
        LoadBool("roundToPowerOfTwo", &roundToPowerOfTwo, FALSE);
        LoadFloat("blackmanCoeff", &blackmanCoeff, 0.42);
        LoadBool("snipEdges", &snipEdges, TRUE);
        LoadBool("allowDownsample", &allowDownsample, FALSE);
        LoadBool("allowUpsample", &allowUpsample, FALSE);
        LoadInt("maxFeatureVectors", &maxFeatureVectors, -1);
        LoadInt("torchPaddingLength", &torchPaddingLength, 200);
        LoadString("padMod", padMod, "reflect");

        LoadInt("numBins", &numBins, 80);
        LoadFloat("lowFreq", &lowFreq, 20.0);
        LoadFloat("highFreq", &highFreq, 0.0);
        LoadFloat("vtlnLow", &vtlnLow, 100.0);
        LoadFloat("vtlnHigh", &vtlnHigh, -500.0);
        LoadBool("debugMel", &debugMel, FALSE);
        LoadBool("htkMode", &htkMode, FALSE);
        LoadString("customFilter", customFilter, "../mel.csv");
    }

} /* end of the s2t namespace */