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
  * Well, it is kind of smoothly.
  *
  * $Created by: HE Erfeng (heerfeng1023@gmail.com) 2023-09
  */


#include <algorithm>
#include <cstdio>
#include <limits>
#include <sstream>
#include <vector>

#include "utils.h"
#include "WaveLoader.h"
#include <stdint.h>
#include "../niutensor/tensor/XTensor.h"
#include <iostream>
using namespace nts;

namespace s2t {
    /*
        there are many oprations about SWAP, they are only used for RIFX
    */
    // A utility class for reading wave header.
    struct WaveHeaderReadGofer {
        std::istream& is;
        bool swap;
        char tag[5];

        WaveHeaderReadGofer(std::istream& is) : is(is), swap(false) {
            memset(tag, '\0', sizeof tag);
        }

        /*
            Should be RIFF
        */
        void Expect4ByteTag(const char* expected) {
            is.read(tag, 4);
            if (is.fail())
                std::cout << "ERROR: WaveData: expected " << expected
                << ", failed to read anything";
            if (strcmp(tag, expected))
                std::cout << "ERROR: WaveData: expected " << expected << ", got " << tag;
        }

        void Read4ByteTag() {
            is.read(tag, 4);
            if (is.fail())
                std::cout << "ERROR: WaveData: expected 4-byte chunk-id, got read error";
        }

        UINT32 ReadUint32() {
            union {
                char result[4];
                UINT32 ans;
            } u;
            is.read(u.result, 4);
            if (swap)
                SWAP4(u.result);
            if (is.fail())
                std::cout << "ERROR: WaveData: unexpected end of file or read error";
            return u.ans;
        }

        UINT16 ReadUint16() {
            union {
                char result[2];
                UINT16 ans;
            } u;
            is.read(u.result, 2);
            if (swap)
                SWAP2(u.result);
            if (is.fail())
                std::cout << "ERROR: WaveData: unexpected end of file or read error";
            return u.ans;
        }
    };

    static void WriteUint32(std::ostream& os, INT32 i) {
        union {
            char buf[4];
            int i;
        } u;
        u.i = i;

        /*
            this place is used to judge big & little endian
        */
#ifdef __BIG_ENDIAN__
        SWAP4(u.buf);
#endif
        os.write(u.buf, 4);
        if (os.fail())
            std::cout << "ERROR: WaveData: error writing to stream.";
    }

    static void WriteUint16(std::ostream& os, INT16 i) {
        union {
            char buf[2];
            INT16 i;
        } u;
        u.i = i;
#ifdef __BIG_ENDIAN__
        SWAP2(u.buf);
#endif
        os.write(u.buf, 2);
        if (os.fail())
            std::cout << "ERROR: WaveData: error writing to stream.";
    }

    void WaveInfo::Read(std::istream& is) {
        WaveHeaderReadGofer reader(is);
        reader.Read4ByteTag();
        if (strcmp(reader.tag, "RIFF") == 0)
            reverse_bytes_ = false;
        else if (strcmp(reader.tag, "RIFX") == 0)
            reverse_bytes_ = true;
        else
            std::cout << "ERROR: WaveData: expected RIFF or RIFX, got " << reader.tag;

#ifdef __BIG_ENDIAN__
        reverse_bytes_ = !reverse_bytes_;
#endif
        reader.swap = reverse_bytes_;

        UINT32 riff_chunk_size = reader.ReadUint32();
        reader.Expect4ByteTag("WAVE");

        UINT32 riff_chunk_read = 0;
        riff_chunk_read += 4;  // WAVE included in riff_chunk_size.

        // Possibly skip any RIFF tags between 'WAVE' and 'fmt '.
        // Apple devices produce a filler tag 'JUNK' for memory alignment.
        reader.Read4ByteTag();
        riff_chunk_read += 4;
        while (strcmp(reader.tag, "fmt ") != 0) {
            UINT32 filler_size = reader.ReadUint32();
            riff_chunk_read += 4;
            for (UINT32 i = 0; i < filler_size; i++) {
                is.get(); // read 1 byte,
            }
            riff_chunk_read += filler_size;
            // get next RIFF tag,
            reader.Read4ByteTag();
            riff_chunk_read += 4;
        }

        ASSERT(strcmp(reader.tag, "fmt ") == 0);
        UINT32 subchunk1_size = reader.ReadUint32();
        UINT16 audio_format = reader.ReadUint16();
        num_channels_ = reader.ReadUint16();
        UINT32 sample_rate = reader.ReadUint32(),
            byte_rate = reader.ReadUint32(),
            block_align = reader.ReadUint16(),
            bits_per_sample = reader.ReadUint16();
        samp_freq_ = static_cast<float>(sample_rate);

        UINT32 fmt_chunk_read = 16;
        if (audio_format == 1) {
            if (subchunk1_size < 16) {
                std::cout << "WaveData: expect PCM format data to have fmt chunk "
                    << "of at least size 16.";
            }
        }
        else if (audio_format == 0xFFFE) {  // WAVE_FORMAT_EXTENSIBLE
            UINT16 extra_size = reader.ReadUint16();
            if (subchunk1_size < 40 || extra_size < 22) {
                std::cout << "WaveData: malformed WAVE_FORMAT_EXTENSIBLE format data.";
            }
            reader.ReadUint16();  // Unused for PCM.
            reader.ReadUint32();  // Channel map: we do not care.
            UINT32 guid1 = reader.ReadUint32(),
                guid2 = reader.ReadUint32(),
                guid3 = reader.ReadUint32(),
                guid4 = reader.ReadUint32();
            fmt_chunk_read = 40;

            // Support only KSDATAFORMAT_SUBTYPE_PCM for now. Interesting formats:
            // ("00000001-0000-0010-8000-00aa00389b71", KSDATAFORMAT_SUBTYPE_PCM)
            // ("00000003-0000-0010-8000-00aa00389b71", KSDATAFORMAT_SUBTYPE_IEEE_FLOAT)
            // ("00000006-0000-0010-8000-00aa00389b71", KSDATAFORMAT_SUBTYPE_ALAW)
            // ("00000007-0000-0010-8000-00aa00389b71", KSDATAFORMAT_SUBTYPE_MULAW)
            if (guid1 != 0x00000001 || guid2 != 0x00100000 ||
                guid3 != 0xAA000080 || guid4 != 0x719B3800) {
                std::cout << "WaveData: unsupported WAVE_FORMAT_EXTENSIBLE format.";
            }
        }
        else {
            std::cout << "WaveData: can read only PCM data, format id in file is: "
                << audio_format;
        }

        for (UINT32 i = fmt_chunk_read; i < subchunk1_size; ++i)
            is.get();  // use up extra data.

        if (num_channels_ == 0)
            std::cout << "WaveData: no channels present";
        if (bits_per_sample != 16)
            std::cout << "WaveData: unsupported bits_per_sample = " << bits_per_sample;
        if (byte_rate != sample_rate * bits_per_sample / 8 * num_channels_)
            std::cout << "Unexpected byte rate " << byte_rate << " vs. "
            << sample_rate << " * " << (bits_per_sample / 8)
            << " * " << num_channels_;
        if (block_align != num_channels_ * bits_per_sample / 8)
            std::cout << "Unexpected block_align: " << block_align << " vs. "
            << num_channels_ << " * " << (bits_per_sample / 8);

        riff_chunk_read += 4 + subchunk1_size;
        // size of what we just read, 4 for subchunk1_size + subchunk1_size itself.

        // We support an optional "fact" chunk (which is useless but which
        // we encountered), and then a single "data" chunk.

        reader.Read4ByteTag();
        riff_chunk_read += 4;

        // Skip any subchunks between "fmt" and "data".  Usually there will
        // be a single "fact" subchunk, but on Windows there can also be a
        // "list" subchunk.
        while (strcmp(reader.tag, "data") != 0) {
            // We will just ignore the data in these chunks.
            UINT32 chunk_sz = reader.ReadUint32();
            if (chunk_sz != 4 && strcmp(reader.tag, "fact") == 0)
                std::cout << "WARNING: Expected fact chunk to be 4 bytes long.";
            for (UINT32 i = 0; i < chunk_sz; i++)
                is.get();
            riff_chunk_read += 4 + chunk_sz;  // for chunk_sz (4) + chunk contents (chunk-sz)

            // Now read the next chunk name.
            reader.Read4ByteTag();
            riff_chunk_read += 4;
        }

        ASSERT(strcmp(reader.tag, "data") == 0);
        UINT32 data_chunk_size = reader.ReadUint32();
        riff_chunk_read += 4;

        // Figure out if the file is going to be read to the end. Values as
        // observed in the wild:
        bool is_stream_mode =
            riff_chunk_size == 0
            || riff_chunk_size == 0xFFFFFFFF
            || data_chunk_size == 0
            || data_chunk_size == 0xFFFFFFFF
            || data_chunk_size == 0x7FFFF000;  // This value is used by SoX.

        if (is_stream_mode)
            std::cout << "Read in RIFF chunk size: " << riff_chunk_size
            << ", data chunk size: " << data_chunk_size
            << ". Assume 'stream mode' (reading data to EOF).";

        if (!is_stream_mode
            && std::abs(static_cast<INT64>(riff_chunk_read) +
                static_cast<INT64>(data_chunk_size) -
                static_cast<INT64>(riff_chunk_size)) > 1) {
            // We allow the size to be off by one without warning, because there is a
            // weirdness in the format of RIFF files that means that the input may
            // sometimes be padded with 1 unused byte to make the total size even.
            std::cout << "WARNING: Expected " << riff_chunk_size << " bytes in RIFF chunk, but "
                << "after first data block there will be " << riff_chunk_read
                << " + " << data_chunk_size << " bytes "
                << "(we do not support reading multiple data chunks).";
        }

        if (is_stream_mode)
            samp_count_ = -1;
        else
            samp_count_ = data_chunk_size / block_align;
    }

    void WaveData::Read(std::istream& is) {
        const UINT32 kBlockSize = 1024 * 1024;

        WaveInfo header;
        header.Read(is);
        int* dimSize = new int[2];
        data_.Resize(2, dimSize, X_FLOAT, 1.0);  // clear the data.
        delete[] dimSize;
        data_.SetZeroAll();
        samp_freq_ = header.SampFreq();

        XTensor buffer, temp;
        UINT32 bytes_to_go = header.IsStreamed() ? kBlockSize : header.DataBytes();
        
        // Once in a while header.DataBytes() will report an insane value;
        // read the file to the end
        while (is && bytes_to_go > 0) {
            UINT32 block_bytes = min(bytes_to_go, kBlockSize);
            UINT32 offset = buffer.GetSize();

            /*std::vector<char> buf;
            offset = buf.size();
            buf.resize(offset + block_bytes);
            is.read(&buf[offset], block_bytes);
            UINT32 bytes_read = is.gcount();
            buf.resize(offset + bytes_read);
            UINT16* data_ptr = reinterpret_cast<UINT16*>(&buf[0]);
            INT16 k = *data_ptr++;*/

            int arraySize = offset + block_bytes;
            int* initDimSize = &(arraySize);
            buffer.Resize(1, initDimSize, X_FLOAT, 1.0);
            temp.Resize(1, initDimSize, X_FLOAT, 1.0);
            int index[1] = { offset };
            is.read(reinterpret_cast<char*>(temp.GetCell(index, 1)), block_bytes);
            UINT32 bytes_read = is.gcount();
            //!!!!!!!!!!!!!!!!!!
            arraySize = offset + bytes_read;
            int * dimSize = &(arraySize);
            buffer.Resize(1, dimSize, X_FLOAT, 1.0);
            //!!!!!!!!!!!!!!!!!!
            buffer.SetData(temp.GetCell(index, 1), buffer.GetDim(0), 0);
            if (!header.IsStreamed())
                bytes_to_go -= bytes_read;
        }

        if (is.bad())
            std::cout << "WaveData: file read error";

        if (buffer.GetDim(0) == 0)
            std::cout << "WaveData: empty file (no data)";

        if (!header.IsStreamed() && buffer.GetDim(0) < header.DataBytes()) {
            std::cout << "WARNING: Expected " << header.DataBytes() << " bytes of wave data, "
                << "but read only " << buffer.GetDim(0) << " bytes. "
                << "Truncated file?";
        }
        int index[1] = { 0 };
        UINT16* data_ptr = reinterpret_cast<UINT16*>(buffer.GetCell(index, 1));

        // The matrix is arranged row per channel, column per sample.
        int tensorSize[] = { buffer.GetDim(0) / header.BlockAlign() };
        /*
            data_ will have two dim, the size of dim0 is the num of channel(should be 1 in ASR)
            and the size of dim1 is the num of sample.
        */
        int realDimSize[] = {buffer.GetDim(0) / header.BlockAlign()};
        data_.Resize(header.NumChannels(), realDimSize, X_FLOAT, 1.0);
        for (UINT32 i = 0; i < data_.GetDim(0); ++i) {
            //for (UINT32 j = 0; j < data_.GetDim(0); ++j) {
            INT16 k = *data_ptr++;
            
            if (header.ReverseBytes())
                SWAP2(k);
            //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            float torchK = k / 3333.3333333333333333333333333333333333333333;
            data_.Set1D(torchK, i);
                //data_[j, i] = k;
                //data_(j, i) = k;
            //}
        }
    }


    // Write 16-bit PCM.

    // note: the WAVE chunk contains 2 subchunks.
    //
    // subchunk2size = data.NumRows() * data.NumCols() * 2.


    void WaveData::Write(std::ostream& os) const {
        os << "RIFF";
        if (data_.GetDim(0) == 0)
            std::cout << "Error: attempting to write empty WAVE file";

        INT32 num_chan = data_.GetDim(0),
            num_samp = data_.GetDim(1),
            bytes_per_samp = 2;

        INT32 subchunk2size = (num_chan * num_samp * bytes_per_samp);
        INT32 chunk_size = 36 + subchunk2size;
        WriteUint32(os, chunk_size);
        os << "WAVE";
        os << "fmt ";
        WriteUint32(os, 16);
        WriteUint16(os, 1);
        WriteUint16(os, num_chan);
        ASSERT(samp_freq_ > 0);
        WriteUint32(os, static_cast<INT32>(samp_freq_));
        WriteUint32(os, static_cast<INT32>(samp_freq_) * num_chan * bytes_per_samp);
        WriteUint16(os, num_chan * bytes_per_samp);
        WriteUint16(os, 8 * bytes_per_samp);
        os << "data";
        WriteUint32(os, subchunk2size);

        /*const float* data_ptr = data_.Data();
        INT32 stride = data_.Stride();*/
        int num_clipped = 0;
        for (INT32 i = 0; i < num_samp; i++) {
            for (INT32 j = 0; j < num_chan; j++) {
                INT32 elem = static_cast<INT32>(trunc(data_.Get2D(j, i)));
                INT16 elem_16 = static_cast<INT16>(elem);
                if (elem < INT_MIN) {
                    elem_16 = INT_MIN;
                    ++num_clipped;
                }
                else if (elem > INT_MAX) {
                    elem_16 = INT_MAX;
                    ++num_clipped;
                }
#ifdef __BIG_ENDIAN__
                SWAP2(elem_16);
#endif
                os.write(reinterpret_cast<char*>(&elem_16), 2);
            }
        }
        if (os.fail())
            std::cout << "Error writing wave data to stream.";
        if (num_clipped > 0)
            std::cout << "WARNING: clipped " << num_clipped
            << " samples out of total " << num_chan * num_samp
            << ". Reduce volume?";
    }


}  // end namespace s2t
