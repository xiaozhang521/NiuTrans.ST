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
    S2TConfig::S2TConfig(int argc, const char** argv)
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


        showConfig();
        /* parse configuration in args */
        model.Load(argsNum, (const char **)args);
        common.Load(argsNum, (const char **)args);
        training.Load(argsNum, (const char **)args);
        inference.Load(argsNum, (const char **)args);

        showConfig();

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
        model.showConfig();
    }

    /* load s2t model configuration from the command */
    void S2TModelConfig::Load(int argsNum, const char** args)
    {
        Create(argsNum, args);

        LoadBool("bigatt", &useBigAtt, false);
        LoadBool("encprenorm", &encPreLN, true);
        LoadBool("decprenorm", &decPreLN, true);
        LoadBool("encl1norm", &encoderL1Norm, false);
        LoadBool("decl1norm", &decoderL1Norm, false);
        LoadBool("decoderonly", &decoderOnly, false);
        LoadBool("enchistory", &useEncHistory, false);
        LoadBool("dechistory", &useDecHistory, false);
        LoadBool("encfinalnorm", &encFinalNorm, true);
        LoadBool("decfinalnorm", &decFinalNorm, true);
        LoadBool("shareencdec", &shareEncDecEmb, false);
        LoadBool("sharedec", &shareDecInputOutputEmb, false);

        LoadInt("fbank", &fbank, 80);
        LoadInt("pad", &pad, -1);
        LoadInt("sos", &sos, -1);
        LoadInt("eos", &eos, -1);
        LoadInt("unk", &unk, -1);
        LoadInt("encemb", &encEmbDim, 512);
        LoadInt("decemb", &decEmbDim, 512);
        LoadInt("maxsrc", &maxSrcLen, 200);
        LoadInt("maxtgt", &maxTgtLen, 200);
        LoadInt("enclayer", &encLayerNum, 6);
        LoadInt("declayer", &decLayerNum, 6);
        LoadInt("maxrp", &maxRelativeLength, -1);
        LoadInt("encffn", &encFFNHiddenDim, 1024);
        LoadInt("decffn", &decFFNHiddenDim, 1024);
        LoadInt("srcvocabsize", &srcVocabSize, -1);
        LoadInt("tgtvocabsize", &tgtVocabSize, -1);
        LoadInt("encheads", &encSelfAttHeadNum, 4);
        LoadInt("decheads", &decSelfAttHeadNum, 4);
        LoadInt("encdecheads", &encDecAttHeadNum, 4);

        LoadFloat("dropout", &dropout, 0.3F);
        LoadFloat("ffndropout", &ffnDropout, 0.1F);
        LoadFloat("attdropout", &attDropout, 0.1F);
    }

    void S2TModelConfig::showConfig()
    {
        /* 19 integers */
        cout << "    - Integer" << endl;
        cout << "\t| " << "fbank = " << fbank <<endl;
        cout << "\t| " << "encEmbDim = " << encEmbDim <<endl;
        cout << "\t| " << "encLayerNum = " << encLayerNum <<endl;
        cout << "\t| " << "encSelfAttHeadNum = " << encSelfAttHeadNum <<endl;
        cout << "\t| " << "encFFNHiddenDim = " << encFFNHiddenDim <<endl;
        cout << "\t| " << "decEmbDim = " << decEmbDim <<endl;
        cout << "\t| " << "decLayerNum = " << decLayerNum <<endl;
        cout << "\t| " << "decSelfAttHeadNum = " << decSelfAttHeadNum <<endl;
        cout << "\t| " << "encDecAttHeadNum = " << encDecAttHeadNum <<endl;
        cout << "\t| " << "decFFNHiddenDim = " << decFFNHiddenDim <<endl;
        cout << "\t| " << "maxRelativeLength = " << maxRelativeLength <<endl;
        cout << "\t| " << "maxSrcLen = " << maxSrcLen <<endl;
        cout << "\t| " << "maxTgtLen = " << maxTgtLen <<endl;
        cout << "\t| " << "sos = " << sos <<endl;
        cout << "\t| " << "eos = " << eos <<endl;
        cout << "\t| " << "pad = " << pad <<endl;
        cout << "\t| " << "unk = " << unk <<endl;
        cout << "\t| " << "srcVocabSize = " << srcVocabSize <<endl;
        cout << "\t| " << "tgtVocabSize = " << tgtVocabSize <<endl;

        cout << "    - Bool" << endl;
        cout << "\t| " << "encoderL1Norm = " << encoderL1Norm <<endl;
        cout << "\t| " << "decoderL1Norm = " << decoderL1Norm <<endl;
        cout << "\t| " << "useBigAtt = " << useBigAtt <<endl;
        cout << "\t| " << "decoderOnly = " << decoderOnly <<endl;
        cout << "\t| " << "encFinalNorm = " << encFinalNorm <<endl;
        cout << "\t| " << "decFinalNorm = " << decFinalNorm <<endl;
        cout << "\t| " << "encPreLN = " << encPreLN <<endl;
        cout << "\t| " << "decPreLN = " << decPreLN <<endl;
        cout << "\t| " << "useEncHistory = " << useEncHistory <<endl;
        cout << "\t| " << "useDecHistory = " << useDecHistory <<endl;
        cout << "\t| " << "shareEncDecEmb = " << shareEncDecEmb <<endl;
        cout << "\t| " << "shareDecInputOutputEmb = " << shareDecInputOutputEmb <<endl;

    }   

} /* end of the s2r namespace */