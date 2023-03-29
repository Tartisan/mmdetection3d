#!/usr/bin/env python3
'''
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
Author: DouZSh(douzesheng@baidu.com)
Date: 2021-07-15 10:51:34
Description: calibrator class define
'''
import os
import tarfile
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class CPDetEntropyCalibrator(trt.IInt8EntropyCalibrator):
    """Calibrator for tensorrt.

    Args:
        shape : image shape get from calibrator, NCHW.
        file_path : image file list path with absolute path per line inside.
        cache_file : load and save from cache_file. Cache file will not generated if exits.
    """
    def __init__(self, shape, input_file_list, cache_file="model.calib"):
        trt.IInt8EntropyCalibrator.__init__(self)
        self.cache_file = cache_file
        self.batch_size, self.channel, self.height, self.width = shape
        self.batch_idx = 0

        with open(input_file_list, "r") as filelist_fp:
            all_lines = filelist_fp.readlines()
            self.files = [s.strip() for s in all_lines]

        self.imgs = self.files
        self.max_batch_idx = len(self.files)//self.batch_size
        self.MAX_POINT_CLOUR_NUM = 160000
        if self.channel<0:
            self.channel=self.MAX_POINT_CLOUR_NUM
        self.data_size = trt.volume([self.batch_size, self.channel, self.height, self.width]) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def next_batch(self):
        """get next batch from calibrator
        """
        if self.batch_idx < self.max_batch_idx:
            batch_imgs = []
            try:
                for file_name in self.files[
                        self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]:
                    if file_name.find(".tgz")>-1:
                        t = tarfile.open(file_name, "r:gz")
                        fp = t.extractfile(t.getnames()[0])
                        data = np.frombuffer(fp.read(), dtype=np.float32)
                    else:
                        with open(file_name, "rb") as fp:
                            data = np.fromfile(fp, dtype=np.float32)
                    data = data.reshape((-1, self.height, self.width))
                    if self.channel==self.MAX_POINT_CLOUR_NUM:
                        print("original pcd shape:{}".format(data.shape))
                        if data.shape[0]>self.MAX_POINT_CLOUR_NUM:
                            data = data[:self.MAX_POINT_CLOUR_NUM]
                        else:
                            data = np.pad(data, ((0, self.MAX_POINT_CLOUR_NUM-data.shape[0]),(0, 0),(0, 0)), 'symmetric')
                    batch_imgs.append(data)
                self.batch_idx += 1
                print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
                return np.ascontiguousarray(batch_imgs)
            except:
                print("error with idx:{} and file:{}".format(self.batch_idx, self.files[
                        self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]))
                fake_data = np.zeros((1, self.channel, self.height, self.width), dtype=np.float32)
                self.batch_idx += 1
                print("continue with fake 0 data:{}".format(fake_data.shape))
                return np.ascontiguousarray(fake_data)
        else:
            return np.array([])

    def get_batch_size(self):
        """get batch size
        """
        return self.batch_size

    def get_batch(self, names):
        """get batch images.
           This is the main function with Calibrator.
           names is useless unless in debug mode.
        """
        print("all image:{} with idx:{}".format(len(self.imgs), self.batch_idx))
        try:
            batch_imgs = self.next_batch()
            print("get image shape:{}".format(batch_imgs.shape))
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size*self.channel*self.height*self.width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))
            return [int(self.device_input)]
        except:
            return None

    def read_calibration_cache(self):
        """If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as file_pointer:
                return file_pointer.read()

    def write_calibration_cache(self, cache):
        """write the result to file.
        """
        with open(self.cache_file, "wb") as file_pointer:
            file_pointer.write(cache)
