import tensorrt as trt
import sys
import os



def get_engine(onnx_file_path="", engine_file_path="", fp16_mode=True, int8_mode=False, save_engine=True,max_batch_size=1, in_ch = 90):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        logger = trt.Logger(trt.Logger.VERBOSE)
        EXPLICIT_BATCH = []
        EXPLICIT_BATCH.append(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        with trt.Builder(logger) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, logger) as parser, builder.create_builder_config() as config:
            config.max_workspace_size = 4194304
            builder.max_batch_size = max_batch_size
            if fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)

            profile = builder.create_optimization_profile()

            # ! mm input     
            # profile.set_shape("input", (1, in_ch, 756, 1008),(1, in_ch, 756, 1008),(1, in_ch, 756, 1008))

            # # ! refine input     
            profile.set_shape("input", (1, in_ch, 756, 1008),(1, in_ch, 756, 1008),(1, in_ch, 756, 1008))

            # # ! nerf input: input 63, input_dir 27
            # profile.set_shape("input", (max_batch_size, in_ch[0]),(max_batch_size, in_ch[0]),(max_batch_size, in_ch[0]))
            # profile.set_shape("input_dir", (max_batch_size, in_ch[1]),(max_batch_size, in_ch[1]),(max_batch_size, in_ch[1]))

            config.add_optimization_profile(profile)

            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            engine = builder.build_engine(network, config)
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    return build_engine(max_batch_size, save_engine)


if __name__ == '__main__':
    # ONNX_NAME = 'logs_minmax/fern_epinerf_trt/nerf.onnx'
    # TRT_NAME = 'logs_minmax/fern_epinerf_trt/nerf_fp16.trt'
    # get_engine(onnx_file_path=ONNX_NAME,engine_file_path=TRT_NAME, max_batch_size=756*1008*4, in_ch=[63,27], fp16_mode=True)

    # ONNX_NAME = 'logs_minmax/fern_epinerf_trt/min_max_ray_net.onnx'
    # TRT_NAME = 'logs_minmax/fern_epinerf_trt/min_max_ray_net_fp16.trt'
    # get_engine(onnx_file_path=ONNX_NAME,engine_file_path=TRT_NAME, max_batch_size=1, in_ch=96, fp16_mode=True)

    ONNX_NAME = 'logs_minmax/fern_epinerf_trt/model_refine.onnx'
    TRT_NAME = 'logs_minmax/fern_epinerf_trt/model_refine_fp16.trt'
    get_engine(onnx_file_path=ONNX_NAME,engine_file_path=TRT_NAME, max_batch_size=1, in_ch=24, fp16_mode=True)