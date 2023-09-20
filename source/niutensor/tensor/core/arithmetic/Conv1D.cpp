/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northeastern University.
* All rights reserved.
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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-08-01
 * $Updated by: Yuhao Zhang (email: yoohao.zhang@gmail.com) 2023-09-20
 */

#include "../../XTensor.h"
#include "../../XName.h"
#include "../../XUtility.h"
#include "Conv1D.h"
#include "cudnn.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
* 
*/
void _Conv1DBase(const XTensor *input, const XTensor *weight, const XTensor *bias, XTensor *output,
                      int stride, int padding, bool useBias)
{
    CheckNTErrors((input->GetDim(0) == output->GetDim(0) &&
                    input->GetDim(1) == weight->GetDim(1) &&
                    output->GetDim(1) == weight->GetDim(0)),
                  "Unmatched dimension");

    CheckDev(input->devID, weight->devID);
    if (useBias) {
        CheckNTErrors(bias->unitNum == weight->GetDim(0), "Unmatched dimension of bias")
        CheckDev(weight->devID, bias->devID);
    }

#ifdef USE_CUDA
    #ifdef USE_CUDNN
    if (input->devID >= 0 || weight->devID >= 0 || (bias != NULL && bias->devID >= 0)) {
        cudaSetDevice(input->devID);
        cudnnHandle_t cudnnHandle;
        CheckCUDNN(cudnnCreate(&cudnnHandle));
        cudnnConvolutionDescriptor_t convDesc;
        cudnnCreateConvolutionDescriptor(&convDesc);

        cudnnTensorDescriptor_t inputDesc, outputDesc;
        CheckCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
        /*CheckCUDNN(cudnnSetTensor4dDescriptor(input_desc,
                                              CUDNN_TENSOR_NCHW,
                                              CUDNN_DATA_FLOAT,
                                              batch_size_,
                                              group_ * in_channel_,
                                              height_,
                                              width_));*/
        cudnnCreateTensorDescriptor(&outputDesc);

        cudnnFilterDescriptor_t filterDesc;
        cudnnCreateFilterDescriptor(&filterDesc);

        // Step 4: Allocate Memory for Buffers
        // Allocate memory for input, output, and workspace buffers on the GPU

        // Step 5: Perform Convolution
        //cudnnConvolutionForward(cudnn_handle, &alpha, inputDesc, inputData, filterDesc, filterData, convDesc, algo, workspace, workspaceSize, &beta, outputDesc, outputData);

        // Step 6: Post-processing (if needed)

        // Step 7: Clean Up Resources
        // Release descriptors, tensors, and GPU memory

        // Destroy the cudnnHandle when done
        cudnnDestroy(cudnnHandle);
        return;
    }
    else {
        // TODO!!
        ShowNTErrors("TODO!");
    }
    #endif
    ShowNTErrors("TODO!");
#endif

    int blockNum = input->GetDim(0);
    int inChannel = input->GetDim(1);
    int outChannel = output->GetDim(1);
    int inputLength = input->GetDim(2);
    int kernelSize = weight->GetDim(2);
    int blockStride = output->GetDim(2);
    int inputBlockSize = input->GetDim(1) * input->GetDim(2);
    int outputBlockSize = output->GetDim(1) * output->GetDim(2);


    DTYPE * ip = (DTYPE*)input->data;
    DTYPE * op = (DTYPE*)output->data;
    DTYPE * wp = (DTYPE*)weight->data;
    DTYPE * bp = NULL;
    if (useBias)
        bp = (DTYPE*)bias->data;

    DTYPE value;
    for (int i = 0; i < blockNum; i++) {
        DTYPE * tmpIp = ip + i * inputBlockSize;
        DTYPE * tmpOp = op + i * outputBlockSize;

//        for (int j = 0, index=-padding; j < blockStride; j++, index += stride) {
//            for (int k = 0; k < outChannel; k++) {
//                DTYPE * tmpWp = wp + k * inChannel * kernelSize;
//
//                value = 0;
//                int tmpIndex = index;
//                for (int x = 0; x < kernelSize; x++) {
//                    for (int y = 0; y < inChannel; y++) {
//                        if (tmpIndex >= 0 && tmpIndex < inputLength)
//                            value += tmpIp[y * inputLength + tmpIndex] * tmpWp[y * kernelSize + x];
//                    }
//                tmpIndex++;
//                }
//
//                if (bp != NULL)
//                    value += bp[k];
//                tmpOp[k * blockStride + j] = value;
//            }
//        }

        for (int j = 0; j < blockStride; j++) {
            for (int k = 0; k < outChannel; k++) {
                DTYPE * tmpWp = wp + k * inChannel * kernelSize;

                value = 0;
                int index = j * stride - padding;

                for (int x = 0; x < kernelSize; x++) {
                    for (int y = 0; y < inChannel; y++) {
                        if (index >= 0 && index < inputLength)
                            value += tmpIp[y * inputLength + index] * tmpWp[y * kernelSize + x];
                    }
                    index++;
                }

                if (bp != NULL)
                    value += bp[k];
                tmpOp[k * blockStride + j] = value;

            }
        }
    }
}

/*
TODO
*/
XTensor Conv1DBias(const XTensor &input, const XTensor &weight, const XTensor &bias,
                   int stride, int padding, bool useBias)
{
    CheckNTErrors((input.order==3 && weight.order==3), "The orders of input and weight should be 3");
    
    XTensor output;

    int outputLength = (input.GetDim(2) + 2 * padding - weight.GetDim(2) / stride) + 1;

    InitTensor3D(&output, input.GetDim(0), weight.GetDim(0), outputLength, X_FLOAT, input.devID);
    output.SetTMPFlag();

    _Conv1DBase(&input, &weight, &bias, &output, stride, padding, useBias);

    /* tensor connections */
    TensorList params(2);
    params.Add((XTensor*)&weight);
    if (useBias)
        params.Add((XTensor*)&bias);

    if (input.enableGrad && output.enableGrad) {
        XLink::MakeLink(&params, &output, MATH_CONV1D);
        XLink::AddParamToHeadInt(&output, stride);
        XLink::AddParamToHeadInt(&output, padding);
    }
    return output;

}

XTensor Conv1DBase(const XTensor& input, const XTensor& weight, int stride, int padding)
{
    CheckNTErrors((input.order == 3 && weight.order == 3), "The orders of input and weight should be 3");

    XTensor output;

    int outputLength = (input.GetDim(2) + 2 * padding - weight.GetDim(2) / stride) + 1;

    InitTensor3D(&output, input.GetDim(0), weight.GetDim(0), outputLength, X_FLOAT, input.devID);
    output.SetTMPFlag();

    _Conv1DBase(&input, &weight, NULL, &output, stride, padding, false);

    /* tensor connections */
    TensorList params(2);
    params.Add((XTensor*)&weight);

    if (input.enableGrad && output.enableGrad) {
        XLink::MakeLink(&params, &output, MATH_CONV1D);
        XLink::AddParamToHeadInt(&output, stride);
        XLink::AddParamToHeadInt(&output, padding);
    }
    return output;

}

} // namespace nts(NiuTrans.Tensor)
