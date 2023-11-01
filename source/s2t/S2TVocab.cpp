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

#include <fstream>
#include <iostream>
#include <locale>
#include "S2TVocab.h"
#include "./S2TConfig.h"

using namespace std;

namespace s2t
{
/* set ids for special tokens */
/* TODO!!! update for multilingual tokens */
void S2TVocab::SetSpecialID(int sos, int eos, int pad, int unk)
{
    sosID = sos;
    eosID = eos;
    padID = pad;
    unkID = unk;
}

/* constructor */
S2TVocab::S2TVocab()
{
    sosID = -1;
    eosID = -1;
    padID = -1;
    unkID = -1;
    vocabSize = -1;
}

void S2TVocab::Load(const string& vocabFN)
{
    string vsz, sid;
    ifstream f(vocabFN, ios::in);
    CheckNTErrors(f.is_open(), "Failed to open the vocabulary file");
        
    /* get the vocab size */
    std::getline(f, vsz);
    // sosID = (int)stol(sid);
    vocabSize = (int)stol(vsz);

    string word, id;
    for (int i = 0; i < vocabSize; i++) {
        // f >> word >> id;
        string line;
        std::getline(f, line);
        //cout << line << endl;
        size_t pos = line.find('\t');
        word = line.substr(0, pos);
        id = line.substr(pos+1, line.length()- (pos + 1));
        //cout << word << " " << id << endl;

        token2id[word] = (int)stol(id);
        id2token[(int)stol(id)] = word;

        //cout << word << " " << id << endl;
    }

    f.close();
}

/* save a vocabulary to a file */
void S2TVocab::Save(const string& vocabFN)
{
    ofstream f(vocabFN, ios::out);

    /* the first line: size of the vocab and the start id */
    f << vocabSize << "\t" << sosID;

    /* other lines: words and indices */
    for (const auto& p : token2id)
        f << p.first << "\t" << p.second;

    f.close();
}

/*
copy data from another vocabulary
>> v - the target vocabulary
*/
void S2TVocab::CopyFrom(const S2TVocab& v)
{
    for (const auto& w2i : v.token2id)
        token2id.insert(w2i);

    for (const auto& i2w : v.id2token)
        id2token.insert(i2w);
}

void S2TVocab::ShowVocab()
{
    for (int i = 0; i < vocabSize; i++) {
        cout << id2token[i] << "\t" << i << endl;
    }
    cout << "Vocab size: " << vocabSize << endl;
}

void S2TVocab::Test()
{
    // utf-8 decode test
    vector<int> tokensId = { 50364, 2664, 1530, 223, 17665, 4510, 104, 19488, 13545, 14812, 6866, 19563, 5157, 50536 };
    vector<string> tokensUtf8;
    cout << "ids" << "-->" << "tokens" << endl;
    for (int i = 0; i < tokensId.size(); i++) {
        string token = id2token[tokensId[i]];
        // token = token.substr(2, token.length() - 2 - 1);
        tokensUtf8.push_back(token);
        cout << tokensId[i] << "-->" << token << endl;
    }
    cout << endl;

    //for (int i = 0; i < tokensUtf8.size(); i++) {
    //   Utf8ToString(tokensUtf8[i]);
    //}
        
    // string testSentence = "�@���Ō�춹�Ԫ1654��";
    // cout << "Origin String: " << testSentence << endl;
    //StringToUtf8(testSentence);
}

wstring S2TVocab::Utf8ToString(const std::vector<unsigned char>& utf8Bytes)
{
    std::wstring result;
    for (size_t i = 0; i < utf8Bytes.size();) {
        wchar_t unicodeChar;
        unsigned char currentByte = utf8Bytes[i++];

        if (currentByte <= 0x7F) {
            unicodeChar = static_cast<wchar_t>(currentByte);
        }
        else if (currentByte <= 0xDF) {
            unicodeChar = static_cast<wchar_t>(currentByte & 0x1F);
            unicodeChar <<= 6;
            unicodeChar |= static_cast<wchar_t>(utf8Bytes[i++] & 0x3F);
        }
        else if (currentByte <= 0xEF) {
            unicodeChar = static_cast<wchar_t>(currentByte & 0x0F);
            unicodeChar <<= 12;
            unicodeChar |= static_cast<wchar_t>((utf8Bytes[i++] & 0x3F) << 6);
            unicodeChar |= static_cast<wchar_t>(utf8Bytes[i++] & 0x3F);
        }
        else if (currentByte <= 0xF7) {
            unicodeChar = static_cast<wchar_t>(currentByte & 0x07);
            unicodeChar <<= 18;
            unicodeChar |= static_cast<wchar_t>((utf8Bytes[i++] & 0x3F) << 12);
            unicodeChar |= static_cast<wchar_t>((utf8Bytes[i++] & 0x3F) << 6);
            unicodeChar |= static_cast<wchar_t>(utf8Bytes[i++] & 0x3F);
        }
        else {
            unicodeChar = L'?';
        }
        result.push_back(unicodeChar);
    }
    return result;
}

} /* end of the s2t namespace */
