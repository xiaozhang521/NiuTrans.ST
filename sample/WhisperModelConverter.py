import torch
import argparse
import numpy as np
from glob import glob
from struct import pack
import os
# import yaml
from collections import OrderedDict
import copy


def checkByteFile(file, x, Info):
    from struct import pack, unpack
    print("!!!!!!!!! Now is Checking {} Data !!!!!!!!!!".format(Info))
    ## check 
    with open(file, 'rb') as f:
        x_check = unpack("f" * x.numel(), f.read())
        x_check = np.array(x_check)

        print(x_check, x_check.shape)
        x_true = x.contiguous().view(-1).cpu().numpy()
        print(x_true, x_true.shape)
        gap = np.abs(x_true - x_check)
        print("max: {}, min: {}, mean: {}".format(np.max(gap), np.min(gap), np.mean(gap)))
        greather = gap > 1e-3
        positions = np.where(greather)[0]
        print(positions, positions.shape, (positions.shape[0] / x_true.shape[0]) * 100)
    print("--------- Finish Checking {} Data ---------".format(Info))


def get_params_list(config, scale):

    l = []

    if not config['bool']['decoderOnly']:
        # encoder

        # conv1d
        for i in range(1,3):
            l.append('encoder.conv{}.weight'.format(i))
            l.append('encoder.conv{}.bias'.format(i))

        # encoder block
        for i in range(config['int']['encLayerNum']):
            l.append('encoder.blocks.{}.attn.key.weight'.format(i))
            l.append('encoder.blocks.{}.attn.value.weight'.format(i))
            l.append('encoder.blocks.{}.attn.value.bias'.format(i))
            l.append('encoder.blocks.{}.attn.query.weight'.format(i))
            l.append('encoder.blocks.{}.attn.query.bias'.format(i))
            l.append('encoder.blocks.{}.attn.out.weight'.format(i))
            l.append('encoder.blocks.{}.attn.out.bias'.format(i))
            l.append('encoder.blocks.{}.attn_ln.weight'.format(i))
            l.append('encoder.blocks.{}.attn_ln.bias'.format(i))
            l.append('encoder.blocks.{}.mlp.0.weight'.format(i))
            l.append('encoder.blocks.{}.mlp.0.bias'.format(i))
            l.append('encoder.blocks.{}.mlp.2.weight'.format(i))
            l.append('encoder.blocks.{}.mlp.2.bias'.format(i))
            l.append('encoder.blocks.{}.mlp_ln.weight'.format(i))
            l.append('encoder.blocks.{}.mlp_ln.bias'.format(i))
        if config['bool']['encFinalNorm']:
            l.append('encoder.ln_post.weight')
            l.append('encoder.ln_post.bias')

        # decoder block
        for i in range(config['int']['decLayerNum']):
            l.append('decoder.blocks.{}.attn.key.weight'.format(i))
            l.append('decoder.blocks.{}.attn.value.weight'.format(i))
            l.append('decoder.blocks.{}.attn.value.bias'.format(i))
            l.append('decoder.blocks.{}.attn.query.weight'.format(i))
            l.append('decoder.blocks.{}.attn.query.bias'.format(i))
            l.append('decoder.blocks.{}.attn.out.weight'.format(i))
            l.append('decoder.blocks.{}.attn.out.bias'.format(i))
            l.append('decoder.blocks.{}.attn_ln.weight'.format(i))
            l.append('decoder.blocks.{}.attn_ln.bias'.format(i))
            if not config['bool']['decoderOnly']:
                # cross attention
                l.append('decoder.blocks.{}.cross_attn.key.weight'.format(i))
                l.append('decoder.blocks.{}.cross_attn.value.weight'.format(i))
                l.append('decoder.blocks.{}.cross_attn.value.bias'.format(i))
                l.append('decoder.blocks.{}.cross_attn.query.weight'.format(i))
                l.append('decoder.blocks.{}.cross_attn.query.bias'.format(i))
                l.append('decoder.blocks.{}.cross_attn.out.weight'.format(i))
                l.append('decoder.blocks.{}.cross_attn.out.bias'.format(i))
                l.append('decoder.blocks.{}.cross_attn_ln.weight'.format(i))
                l.append('decoder.blocks.{}.cross_attn_ln.bias'.format(i))

            l.append('decoder.blocks.{}.mlp.0.weight'.format(i))
            l.append('decoder.blocks.{}.mlp.0.bias'.format(i))
            l.append('decoder.blocks.{}.mlp.2.weight'.format(i))
            l.append('decoder.blocks.{}.mlp.2.bias'.format(i))
            l.append('decoder.blocks.{}.mlp_ln.weight'.format(i))
            l.append('decoder.blocks.{}.mlp_ln.bias'.format(i))
        if config['bool']['decFinalNorm']:
            l.append('decoder.ln.weight')
            l.append('decoder.ln.bias')

        if not config['bool']['shareEncDecEmb']:
            l.append('decoder.token_embedding.weight')

        l.append('decoder.positional_embedding')

    return l

def get_whisper_parameters(params, config, scale):
    '''
    get flattend transformer model parameters
    '''

    l = get_params_list(config, scale)

    p = []

    for k in params:
        assert k in l, "{} should be include".format(k)

    for name in l:
        name_keys = name.split('.')
        if "weight" in name_keys and ("attn" in name_keys or "cross_attn" in name_keys or "mlp" in name_keys):
            
            print(name, params[name].shape)
            p.append(params[name].t())
        else:
            p.append(params[name])

    return p

def get_whisper_config(config, scale):
    # 12 booleans
    config_dict_bool = OrderedDict({
        "encoderL1Norm": False,
        "decoderL1Norm": False,
        "useBigAtt": False,
        "decoderOnly": False,
        "encFinalNorm": True,
        "decFinalNorm": True,
        "encPreLN": True,
        "decPreLN": True,
        "useEncHistory": False,
        "useDecHistory": False,
        "shareEncDecEmb": False,
        "shareDecInputOutputEmb": True,
    })
    # 19 integers 
    config_dict_int = OrderedDict({
        "srcVocabSize": -1,
        "tgtVocabSize":config['n_vocab'],
        "sos": 50258,
        "eos": 50257,
        "pad": -1,   # 50257
        "unk": -1,
        "maxSrcLen":config['n_audio_ctx'],
        "maxTgtLen":config['n_text_ctx'],
        "maxRelativeLength": -1, 
        "fbank":config['n_mels'],
        "encEmbDim":config['n_audio_state'],
        "encLayerNum":config['n_audio_layer'],
        "encSelfAttHeadNum":config['n_audio_head'],
        "encFFNHiddenDim": scale,
        "decEmbDim":config['n_text_state'],
        "decLayerNum":config['n_text_layer'],
        "decSelfAttHeadNum":config['n_text_head'],
        "encDecAttHeadNum":config['n_text_head'],
        "decFFNHiddenDim": scale,
        "fnnActFunType": 1,
    })
    # 3 floats
    config_dict_float = OrderedDict({
        "dropout": float(0.0),
        "ffnDropout": float(0.0),
        "attDropout": float(0.0),
    })
    
    whisper_config = {"bool":config_dict_bool, "int":config_dict_int, "float":config_dict_float}

    return whisper_config

def main():
    parser = argparse.ArgumentParser(
    description='The model converter for NiuTrans.ST, from Whisper model')
    parser.add_argument('-src', help='The pattern used to find whisper(origin) checkpoints, e.g., \'checkpoint*\'',
                        type=str, default='medium.pt')
    parser.add_argument('-tgt', help='The file name prefix for NiuTrans.ST models',
                        type=str, default='model')
    parser.add_argument('-mode', help='Storage mode, FP32 (Default) or FP16', type=str, default='fp32')
    args = parser.parse_args()
    args.mode = args.mode.lower()
    print(args)

    with torch.no_grad():

        task = "convert"
        # task = "check"

        model_file = args.src
        taget_file = args.tgt
        
        # model_scale
        tiny = 1536
        medium = 4096
        large = 5120

        model_scale = large
        model_name = "large-v2"

        model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "whisper", "model", "{}.pt".format(model_name))
        taget_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper_{}_niutrans_s2t.bin".format(model_name))
       
        if os.path.exists(model_file):

            print("source model: \'{}\'\ntarget model: \'{}\'".format(model_file, taget_file))

            model = torch.load(model_file, map_location='cpu')
            # print(model)

            print("+ checkpoint info: {}".format(model.keys()))
            print("+ Whisper Config")
            if model['dims'] is not None:
                config = model['dims']
                for k in config.keys():
                    if isinstance(config[k], str):
                        config[k] = int(config[k])
                    print("\t- {} : {}".format(k, config[k]))
            
            if model['model_state_dict'] is not None:
                params = model['model_state_dict']

            config = get_whisper_config(config, model_scale)
            print(config , len(config['int'].keys()) + len(config['bool'].keys()) + len(config['float'].keys()))

            origin_params = copy.deepcopy(params)

            # del params['decoder.positional_embedding']
            del params['encoder.positional_embedding']

            params = get_whisper_parameters(params, config, model_scale)
            print("num of params: ", len(params))
            
            if task == "convert":

                print("----- Convert Mode -----")

                bool_config_list = list(config['bool'].values())
                int_config_list = list(config['int'].values())
                float_config_list = list(config['float'].values())
                print(bool_config_list, int_config_list, float_config_list)

                bool_configs = pack('?' * len(bool_config_list), *bool_config_list)
                int_configs = pack('i' * len(int_config_list), *int_config_list)
                float_configs = pack('f' * len(float_config_list), *float_config_list)

                # package
                with open(taget_file, 'wb') as f:
                    # part 1: model configurations
                    f.write(bool_configs)
                    f.write(int_configs)
                    f.write(float_configs)

                    # part 2: values of parameters (in FP32 or FP16)
                    for p in params:
                        if args.mode == 'fp32':
                            values = pack("f" * p.numel(), *
                                        (p.contiguous().view(-1).cpu().numpy()))
                            f.write(values)
                        elif args.mode == 'fp16':
                            values = pack(
                                "e" * p.numel(), *(p.contiguous().view(-1).cpu().numpy().astype(np.float16)))
                            f.write(values)
                
            else:

                print("----- Test Mode -----")
                # print(model['model_state_dict'].keys())
                # test_p = model['model_state_dict']['decoder.token_embedding.weight']
                # print(test_p, test_p.shape)
                # checkByteFile("./pos_output.bin", origin_params['encoder.positional_embedding'], "position embedding")
                i = 0
                #print(origin_params['encoder.blocks.{}.attn_ln.weight'.format(i)])
                checkByteFile("./weight.bin", origin_params['encoder.blocks.{}.attn_ln.weight'.format(i)], "weight")
                #print(origin_params['encoder.blocks.{}.attn_ln.bias'.format(i)])
                checkByteFile("./bias.bin", origin_params['encoder.blocks.{}.attn_ln.bias'.format(i)], "bias")
                


if __name__ == '__main__':
    main()
