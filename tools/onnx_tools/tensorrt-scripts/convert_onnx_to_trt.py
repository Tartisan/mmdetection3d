# -*- coding: UTF-8 -*-
#!/usr/bin/env python3
'''
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
Author: DouZSh(douzesheng@baidu.com)
Date: 2021-07-14 21:35:17
Description: convert onnx model to trt model. BUILD engine.
'''
import os
import argparse
import logging
import tensorrt as trt


logging.basicConfig(level=logging.INFO)


def get_args():
    """modify settings

    Returns:
        dict: arguments dict
    """
    # modify settings
    parser = argparse.ArgumentParser(
        description='Convert onnx/caffe model to TensorRT engine file')
    parser.add_argument('model', help='input ONNX model path')
    parser.add_argument('--proto', default=None,
                        help='input caffe prototxt path, None for ONNX model')
    parser.add_argument('--mode', choices=['int8', 'fp16', 'fp32', 'all'], default='fp32',
                        help='conver model precision setting')
    parser.add_argument('-b', '--max-batch-size', default=64, type=int,
                        help='max batch size for trt engine generated')
    parser.add_argument('-w', '--max-workspace-size', default=2, type=float,
                        help='max workspae size for tactic use. 2GB for default value.')
    parser.add_argument('--dla', default=-1, type=int,
                        help='convert with dla on NVIDIA Xavier.')
    parser.add_argument('-s', '--save', default="result.trt",
                        help='saved path for trt engine')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="print TRT with verbose log mode")
    parser.add_argument('-f', '--fast-mode', action='store_true',
                        help="Use FP16 precision for INT8 mode")
    parser.add_argument('-t', '--strict-type', action='store_true',
                        help="Use strict types for TRT in FP16 | int8 mode")
    parser.add_argument('--fix-shape', action='store_true',
                        help="Use fix shape for engine profiler")
    parser.add_argument('--dataset', choices=['imagenet', 'none', 'trafficlight', 'pointnet', "cpdet"], default='none',
                        help="using trasform and dataset with given dataset.")
    parser.add_argument('--calibrator-image-file',
                        help='calibrator image file list with absolute path inside')
    parser.add_argument('--calibrator-cache', default="model.calib",
                        help='calibrator cache file will load/save.')
    parser.add_argument('--mark-output', nargs="+",
                        default=['prediction'], help="outputs for mark in caffe model.")
    parser.add_argument('--no-cudnn', action='store_true',  help="disable cudnn in tactic.")
    parser.add_argument('--no-cublaslt', action='store_true',  help="disable cublaslt in tactic.")
    return parser.parse_args()


def onnx_to_trt(args):
    ''' convert onnx to tensorrt engine, use mode of ['fp32', 'fp16', 'int8']
    :return: trt engine
    '''
    if args.verbose:
        trt_logger = trt.Logger(trt.Logger.VERBOSE)
    else:
        trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, "")
    # TRT7中的onnx解析器的network，需要指定EXPLICIT_BATCH
    flag = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(trt_logger) as builder, \
            builder.create_network(flag) as network, \
            builder.create_builder_config() as config:
        if args.proto is None:
            parser = trt.OnnxParser(network, trt_logger)
            logging.info(
                'Loading ONNX file from path {}...'.format(args.model))
            with open(args.model, 'rb') as model:
                logging.info('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    for err in range(parser.num_errors):
                        logging.error(parser.get_error(err))
                    raise TypeError("Parser parse failed.")
        else:
            parser = trt.CaffeParser()
            logging.info('Loading caffe model from path {} and {}...'.format(
                args.proto, args.model))
            model_tensors = parser.parse(
                deploy=args.proto,
                model=args.model,
                network=network,
                dtype=trt.float32)
            if not model_tensors:
                for err in range(parser.num_errors):
                    logging.error(parser.get_error(err))
                raise TypeError("Parser parse failed.")
            for mark_output in args.mark_output:
                network.mark_output(model_tensors.find(mark_output))

        # for EXPLICIT_BATCH this is useless
        builder.max_batch_size = args.max_batch_size
        config.max_workspace_size = 1 << 30  # 1GB
        config.max_workspace_size = int(
            config.max_workspace_size*args.max_workspace_size)
        tactic_source = 1 << int(trt.TacticSource.CUBLAS)
        if not args.no_cudnn:
            tactic_source |= 1 << int(trt.TacticSource.CUDNN)
        if not args.no_cublaslt:
            tactic_source |= 1 << int(trt.TacticSource.CUBLAS_LT)
        logging.info("using tactic source:{}.".format(tactic_source))
        config.set_tactic_sources(tactic_source)
        logging.info('config builder for max_batch_size, max_workspace_size:{}'.format(
            config.max_workspace_size))

        # for DLA use
        if args.dla > -1:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            config.DLA_core = args.dla
            config.default_device_type = trt.DeviceType.DLA
            logging.info("config builder for dla core:{}".format(args.dla))

        input_shape = network.get_input(0).shape[1:]
        if args.fix_shape:
            min_input_shape = (args.max_batch_size, ) + input_shape
        else:
            min_input_shape = (1, ) + input_shape
        opt_input_shape = (args.max_batch_size, ) + input_shape
        max_input_shape = (args.max_batch_size, ) + input_shape
        logging.info(
            'config builder for engine min/opt/max shape with input:{}'.format(input_shape))

        logging.info("set builder in mode: {:s}".format(args.mode))
        if args.mode == 'int8':
            assert builder.platform_has_fast_int8, "not support int8"
            config.set_quantization_flag(
                trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
            if args.dla < 0:
                if builder.platform_has_fast_fp16 and args.fast_mode:
                    logging.info("Using FP16 in int8 for more tactic sources.")
                    config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)
            tactic_source = 1 << int(trt.TacticSource.CUBLAS) | 1 << int(
                trt.TacticSource.CUBLAS_LT)
            # config.set_tactic_sources(tactic_source)
            if args.dataset == "cpdet":
                from cpdet_calibrator import CPDetEntropyCalibrator
                calib = CPDetEntropyCalibrator(
                    (args.max_batch_size,
                     input_shape[0],
                     input_shape[1],
                     input_shape[2]),
                    args.calibrator_image_file,
                    cache_file=args.calibrator_cache)
                if input_shape[1] == 9:  # for pfe input with diff channels
                    min_input_shape = (
                        min_input_shape[0],     1, min_input_shape[2], min_input_shape[3])
                    opt_input_shape = (
                        opt_input_shape[0], 160000, opt_input_shape[2], opt_input_shape[3])
                    max_input_shape = (
                        max_input_shape[0], 160000, max_input_shape[2], max_input_shape[3])
                    profile = builder.create_optimization_profile()
                    profile.set_shape(network.get_input(0).name, min=min_input_shape,
                                      opt=opt_input_shape,
                                      max=max_input_shape)
                    config.set_calibration_profile(profile)
            config.int8_calibrator = calib

        elif args.mode == 'fp16':
            assert builder.platform_has_fast_fp16, "not support fp16"
            config.set_flag(trt.BuilderFlag.FP16)

        if args.strict_type:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        logging.info("setting up profile with min,opt,max.")
        profile = builder.create_optimization_profile()
        profile.set_shape(network.get_input(0).name, min=min_input_shape,
                          opt=opt_input_shape,
                          max=max_input_shape)
        config.add_optimization_profile(profile)
        # network.get_input(0).shape = (-1,) + input_shape

        logging.info(
            'Building an engine from file {}; this may take a while...'.format(args.model))
        if int(trt.__version__[0]) < 8:
            engine = builder.build_engine(network, config)
        else:
            engine = builder.build_serialized_network(network, config)
        if engine is None:
            logging.error("engine create failed.")
            return
        logging.info("Created engine success! ")

        # 保存计划文件
        logging.info('Saving TRT engine file to path {}...'.format(args.save))
        with open(args.save, "wb") as engine_file:
            if int(trt.__version__[0]) < 8:
                engine_file.write(engine.serialize())
            else:
                engine_file.write(engine)
        logging.info('Engine file has already saved to {}!'.format(args.save))
        return


def main(args: dict):
    """convertor main entrance
    """
    if args.mode == 'all':
        convert_types = ['fp32', 'fp16', 'int8']
    else:
        convert_types = [args.mode]
    filename, file_extension = os.path.splitext(args.save)
    for mode in convert_types:
        args.save = filename+"-"+mode+file_extension
        args.mode = mode
        try:
            onnx_to_trt(args)
        except Exception as err:
            logging.error("build {} failed with err:{}".format(mode, err))


if __name__ == "__main__":
    main(get_args())
