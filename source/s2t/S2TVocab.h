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
 * $Created by: Yuhao
  Zhang (yoohao.zhang@gmail.com) 2023-09-19
 */

#ifndef __VOCAB_S2T__
#define __VOCAB_S2T__

#include <cstdio>
#include <unordered_map>

using namespace std;

/* the s2t namespace */
namespace s2t {

/* the speech-to-text vocabulary class */
class S2TVocab
{
public:
    /* id of start-of-sequence token */
    int sosID;

    /* id of end-of-sequence token */
    int eosID;

    /* id of paddings */
    int padID;

    /* id of unknown tokens */
    int unkID;

    /* id to indicate there is no time stamp */
    int noTimeStamps;

    /* id to indicate silent */
    int noSpeech;

    /* id to indicate start of some special tokens*/
    int startPrev;

    /* id to indicate start of language model*/
    int startLM;

    /* ids of languages */
    int* langIDs;

    /* ids of time stamp */
    int* tStampIDs;

    /* ids of task */
    int* taskIDs;

    /* size of the vocabulary */
    int vocabSize;

    /* a dict that maps tokens to ids */
    unordered_map<string, int> token2id;

    /* a dict that maps ids to words */
    unordered_map<int, string> id2token;

public:
    /* constructor */
    S2TVocab();

    /* set ids for special tokens */
    void SetSpecialID(int sos, int eos, int pad, int unk);

    /* load a vocabulary from a file */
    void Load(const string& vocabFN);

    /* save a vocabulary to a file */
    void Save(const string& vocabFN);

    /* copy data from another vocab */
    void CopyFrom(const S2TVocab& v);

    void ShowVocab();

    void Test();

};
} /* end of the s2t namespace */



#endif /* __VOCAB_S2T__ */
