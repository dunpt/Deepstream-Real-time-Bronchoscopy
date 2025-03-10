#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import argparse
import sys
sys.path.append('../')

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer
from common.platform_info import PlatformInfo
from common.bus_call import bus_call

import pyds
import cv2


from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import numpy as np
# np.set_printoptions(threshold=np.inf)

from scipy import ndimage
from scipy.ndimage import binary_fill_holes

import os
import os.path
from os import path
import random

from threading import Thread
from multiprocessing import Process

from collections import deque, Counter
import json
# from playsound import playsound
# import pygame

import subprocess
from datetime import datetime
# import pytz

# # Define Vietnam time zone
# vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')

frame_count = {}
saved_count = {}
saved_count["stream_0"] = 0
forward_count = {}
forward_count["Vocal cords"]  = 0
forward_count["Trachea in"]  = 0
forward_count["Trachea out"]  = 0
forward_count["forward_bool"] = 0
forward_count["show_rate"] = 122523452343
forward_count["time_skip"] =  0
forward_count["show_image"]  = 0
forward_count["bool_image"] = 0
forward_count["bubble_mask"] = []
show_rate = 10
sound_delay = {"count" : 1}

mgp_queue_in = deque()
mgp_queue_out = deque()
ttkv_queue = deque()
counts = {i: 0 for i in range(0, 7)}

# global forward_bool
# forward_bool = False

PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_BATCH_TIMEOUT_USEC = 33000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# def sound():
#     print("Start play sound")
#     playsound("pip_new.mp3")
#     print("Finish play sound")

# pygame.init()
# pygame.mixer.pre_init(frequency=48000, buffer=2048)
# pygame.mixer.init()
# detected_sound = pygame.mixer.Sound('/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/pip_sound.mp3')


	

def sgie_src_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list


    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            ### tojson
            # file_path = '/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/transfer.json'
            # with open(file_path, 'r') as file:
            #     data = json.load(file)

            # forward_count["forward_bool"] = data["bool"] 
            ###


            label_list = ['Vocal cords', 'Main carina', 'Intermediate bronchus', 'Right superior lobar bronchus', 'Right inferior lobar bronchus', 'Right middle lobar bronchus', 'Left inferior lobar bronchus', 'Left superior lobar bronchus', 'Right main bronchus', 'Left main bronchus', 'Trachea', 'N/A']

            # print("Vocal cords", forward_count["Vocal cords"])
            # print("Trachea", forward_count["Trachea"])
            # with open('/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/labels_MGPKV.txt', "a") as file:
            #     file.write(" ".join(map(str, label_list[obj_meta.class_id])) + "\n")

            print('Trachea in', forward_count["Trachea in"])
            print('Trachea out', forward_count["Trachea out"])
            print(forward_count["forward_bool"] )

            # if obj_meta.class_id == 0:
            #     forward_count["Vocal cords"] += 1
            if forward_count["forward_bool"] == 0:
                mgp_queue_in.append(obj_meta.class_id)
                if obj_meta.class_id == 10:
                    forward_count["Trachea in"] += 1

                if len(mgp_queue_in) > 100:
                    popped = mgp_queue_in.popleft()
                
                    # if popped == 0:
                    #     forward_count["Vocal cords"] -= 1
                    if popped == 10:
                        forward_count["Trachea in"] -= 1
                
                # in tang kieu len 80/100
                # out 10,15/100
                
                if forward_count["Trachea in"] >= 80: 
                    forward_count["forward_bool"] = 1
                    forward_count["Trachea in"] = 0
                    mgp_queue_in.clear()
            elif forward_count["forward_bool"] == 1:
                mgp_queue_out.append(obj_meta.class_id)
                if obj_meta.class_id == 10:
                    forward_count["Trachea out"] += 1

                if len(mgp_queue_out) > 30:
                    popped = mgp_queue_out.popleft()

                    # if popped == 0:
                    #     forward_count["Vocal cords"] -= 1
                    if popped == 10:
                        forward_count["Trachea out"] -= 1
                
                # in tang kieu len 80/100
                # out 10,15/100
                
                if forward_count["Trachea out"] >= 10: 
                    forward_count["forward_bool"] = 0
                    forward_count["Trachea out"] = 0
                    mgp_queue_out.clear()

                ### tojson
                # file_path = '/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/transfer.json'
                # with open(file_path, 'r') as file:
                #     data = json.load(file)

                # data["bool"] = forward_count["forward_bool"]

                # with open(file_path, 'w', encoding='utf-8') as json_file:
                #     json.dump(data, json_file, ensure_ascii=False, indent=4)
                ###

            # print("Vocal cords", forward_count["Vocal cords"])
            # print("Trachea", forward_count["Trachea"])
            # print("mgp_queue_in", mgp_queue_in)
            # print("forward_bool", forward_count["forward_bool"])

            try: 
                l_obj=l_obj.next
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
            except StopIteration:
                break

            # print("obj_meta.class_id mgp", obj_meta.class_id)
        

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK

def bubbles_gie_src_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
    num_rects=0

    relative_path = '/'.join(stream_path.split('/')[-3:-1]) 
    bubble_mask_path = f'/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/BV Bạch Mai_infered/{relative_path}/bubble_mask_all'
    if not os.path.exists(bubble_mask_path):
        os.makedirs(bubble_mask_path)

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list


    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_meta.rect_params.border_color.red = 0.0
            obj_meta.rect_params.border_color.green = 0.0
            obj_meta.rect_params.border_color.blue = 0.0
            obj_meta.rect_params.border_color.alpha = 0.0

            mask_meta = obj_meta.mask_params

            #position_labels
            array_mask = mask_meta.get_mask_array()
            array_mask = array_mask.reshape(480, 480)    
            array_mask = sigmoid(array_mask)
            threshold = 0.5

            mask_gray_scale = (array_mask > threshold).astype(float) * 255
            mask_gray_scale = mask_gray_scale.astype(np.uint8)
            mask_gray_scale = binary_fill_holes(mask_gray_scale).astype(np.uint8) * 255
            mask_image = Image.fromarray(mask_gray_scale)

            array_mask = (array_mask > threshold).astype(float) * 1
            forward_count["bubble_mask"] = array_mask
            
            mask = array_mask >= 1
            indices = np.argwhere(mask)

            bubble_img_path = "{}/frame_{}.jpg".format(bubble_mask_path, frame_number)
            mask_image.save(bubble_img_path)

            # if (indices.size > 0) and (obj_meta.class_id != 0) and (forward_count["forward_bool"] >= 1) and (saved_count["stream_0"] % int(forward_count["show_rate"])) == 0 and ((iou <= 0.2)) :
            #     #change color
            #     obj_meta.rect_params.border_color.red = 0.0
            #     obj_meta.rect_params.border_color.green = 1.0
            #     obj_meta.rect_params.border_color.blue = 1.0
            #     obj_meta.rect_params.border_color.alpha = 0.0
                
            #     y_offset, x_offset = indices.min(axis=0)
            #     x_offset = x_offset * 790 / 480
            #     y_offset = y_offset * 790 / 480

            #     display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            #     display_meta.num_labels = 1
            #     py_nvosd_text_params = display_meta.text_params[0]

            #     # label_list = ['Niêm mạc xung huyết', 'Có mảng sắc tố đen', 'Hẹp lòng phế quản', 'Cựa phế quản phù nề', 'Niêm mạc thâm nhiễm', 'Tăng sinh mạch', 'Khối u khí phế quản', 'Bình thường' 'N/A']
            #     label_list = ['Bubbles', "No object"]

            #     py_nvosd_text_params.display_text = label_list[obj_meta.class_id]
                
            #     # Now set the offsets where the string should appear
            #     py_nvosd_text_params.x_offset =  x_offset.astype(np.int64)
            #     py_nvosd_text_params.y_offset =  y_offset.astype(np.int64)

            #     # # Font , font-color and font-size
            #     py_nvosd_text_params.font_params.font_name = "Serif"
            #     py_nvosd_text_params.font_params.font_size = 15
            #     # # set(red, green, blue, alpha); set to White
            #     py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            #     # # Text background color
            #     py_nvosd_text_params.set_bg_clr = 1
            #     # # set(red, green, blue, alpha); set to Black
            #     py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            #     # # Using pyds.get_string() to get display_text as string
            #     pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            

            try: 
                l_obj=l_obj.next
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
            except StopIteration:
                break

            # print("obj_meta.class_id mgp", obj_meta.class_id)
        

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK

def pgie_src_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
    num_rects=0

    relative_path = '/'.join(stream_path.split('/')[-3:-1]) 
    folder_path = f'/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/BV Bạch Mai_infered/{relative_path}/masks'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_name = relative_path
    folder_path = f'/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/BV Bạch Mai_infered/{folder_name}/mask'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    label_path = f'/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/BV Bạch Mai_infered/{folder_name}/label'
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    bubble_mask_path = f'/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/BV Bạch Mai_infered/{folder_name}/bubble_mask_all'
    if not os.path.exists(bubble_mask_path):
        os.makedirs(bubble_mask_path)
    
    iou_path = f'/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/BV Bạch Mai_infered/{folder_name}/iou'
    if not os.path.exists(iou_path):
        os.makedirs(iou_path)

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list


    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list

        is_first_obj = True
        save_image = False
        

        # saved_count["stream_{}".format(frame_meta.pad_index)] = 0
        if forward_count["time_skip"] == 150:
            forward_count["show_image"] = 0
            forward_count["time_skip"] = 0
            forward_count["bool_image"] = 0

        if forward_count["bool_image"] == 1:
            forward_count["time_skip"] += 1
            forward_count["show_image"] = 0

        # print("l_obj_1", l_obj)
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # print(obj_meta.class_id, obj_meta.confidence)
            
            ttkv_queue.append(obj_meta.class_id)
            # print("ttkv_queue", ttkv_queue)
            frequency = Counter(ttkv_queue)

            non_zero_frequencies = [count for value, count in frequency.items() if value != 0]
            highest_frequency = max(non_zero_frequencies, default=0)

            if len(ttkv_queue) > 30: 
                popped = ttkv_queue.popleft()
            
            if (highest_frequency >= 20): 
                forward_count["show_rate"] = 1
            else:
                forward_count["show_rate"] = 122523452343
            

            obj_meta.rect_params.border_color.red = 0.0
            obj_meta.rect_params.border_color.green = 0.0
            obj_meta.rect_params.border_color.blue = 0.0
            obj_meta.rect_params.border_color.alpha = 0.0

            # mask_data
            mask_meta = obj_meta.mask_params

            #position_labels
            array_mask = mask_meta.get_mask_array()
            array_mask = array_mask.reshape(480, 480)    
            array_mask = sigmoid(array_mask)
            threshold = 0.5

            mask_gray_scale = (array_mask > threshold).astype(float) * 255
            mask_gray_scale = mask_gray_scale.astype(np.uint8)
            mask_gray_scale = binary_fill_holes(mask_gray_scale).astype(np.uint8) * 255
            mask_image = Image.fromarray(mask_gray_scale)
            
            array_mask = (array_mask > threshold).astype(float) * 1
            
            mask = array_mask >= 1
            indices = np.argwhere(mask)

            # bubble_mask_gray_scale = forward_count["bubble_mask"] * 255
            # bubble_mask_gray_scale = bubble_mask_gray_scale.astype(np.uint8)
            # bubble_mask_gray_scale = binary_fill_holes(bubble_mask_gray_scale).astype(np.uint8) * 255
            # bubble_mask_image = Image.fromarray(bubble_mask_gray_scale)
            bubble_img_path = "{}/frame_{}.jpg".format(bubble_mask_path, frame_number)
            bubble_mask_image = Image.open(bubble_img_path)

            # Calculate IoU
            # intersection = np.logical_and(array_mask, forward_count["bubble_mask"]).sum()
            # union = np.logical_or(array_mask, forward_count["bubble_mask"]).sum()
            # iou = round(intersection / union, 5) if union > 0 else 0

            mask_array = np.array(mask_image)
            bubble_mask_array = np.array(bubble_mask_image)

            # Ensure masks are binarized (0 or 255)
            mask_array = (mask_array > 0).astype(np.uint8)
            bubble_mask_array = (bubble_mask_array > 0).astype(np.uint8)

            # Calculate intersection and union
            intersection = np.logical_and(mask_array, bubble_mask_array).sum()
            union = np.logical_or(mask_array, bubble_mask_array).sum()

            # Calculate IoU
            iou = round(intersection / union, 5) if union > 0 else 0

            # if frame_number == 531:
                # print(type(array_mask), array_mask.shape, array_mask)
                # print(type(forward_count["bubble_mask"]) , forward_count["bubble_mask"].shape, forward_count["bubble_mask"])
                # np.savetxt('array_mask.txt', array_mask)
                # np.savetxt('bubble_mask.txt', forward_count["bubble_mask"])
                # print(intersection)
                # print(union)
                # print(iou)


            if (indices.size > 0) and (obj_meta.class_id != 0) and (forward_count["forward_bool"] >= 1) and (saved_count["stream_0"] % int(forward_count["show_rate"])) == 0 and ((iou <= 0.1)):
                #change color
                obj_meta.rect_params.border_color.red = 0.0
                obj_meta.rect_params.border_color.green = 1.0
                obj_meta.rect_params.border_color.blue = 1.0
                obj_meta.rect_params.border_color.alpha = 0.0
                
                y_offset, x_offset = indices.min(axis=0)
                x_offset = x_offset * 790 / 480
                y_offset = y_offset * 790 / 480

                display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                display_meta.num_labels = 1
                py_nvosd_text_params = display_meta.text_params[0]

                # label_list = ['Niêm mạc xung huyết', 'Có mảng sắc tố đen', 'Hẹp lòng phế quản', 'Cựa phế quản phù nề', 'Niêm mạc thâm nhiễm', 'Tăng sinh mạch', 'Khối u khí phế quản', 'Bình thường' 'N/A']
                label_list = ['Binh thuong', 'Khoi u']

                py_nvosd_text_params.display_text = label_list[obj_meta.class_id]
                
                # Now set the offsets where the string should appear
                py_nvosd_text_params.x_offset =  x_offset.astype(np.int64)
                py_nvosd_text_params.y_offset =  y_offset.astype(np.int64)

                # # Font , font-color and font-size
                py_nvosd_text_params.font_params.font_name = "Serif"
                py_nvosd_text_params.font_params.font_size = 15
                # # set(red, green, blue, alpha); set to White
                py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

                # # Text background color
                py_nvosd_text_params.set_bg_clr = 1
                # # set(red, green, blue, alpha); set to Black
                py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
                # # Using pyds.get_string() to get display_text as string
                pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

                if forward_count["show_image"] == 0 and forward_count["time_skip"] == 0:
                    forward_count["show_image"] = 1
                    forward_count["bool_image"] = 1

                if forward_count["show_image"] == 1 and forward_count["time_skip"] == 0:
                    img_path = "{}/frame_{}.jpg".format(folder_path, frame_number)
                    mask_image.save(img_path)

                    # bubble_img_path = "{}/frame_{}.jpg".format(bubble_mask_path, frame_number)
                    # bubble_mask_image.save(bubble_img_path)

                    txt_label = "{}/frame_{}.txt".format(label_path, frame_number)
                    with open(txt_label, "w") as f:
                        f.write(label_list[obj_meta.class_id] + "\n")

                    txt_iou = "{}/frame_{}.txt".format(iou_path, frame_number)
                    with open(txt_iou, "w") as f:
                        f.write(str(iou) + "\n")


                
                
                sound_delay["count"] += 1
            else:
                
                min_row, min_col = 0,0
            
            try: 
                l_obj=l_obj.next
                # print("obj_meta.class_id", obj_meta.class_id)

                # pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)

                if forward_count["forward_bool"] < 1:
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                elif saved_count["stream_0"] % int(forward_count["show_rate"]) != 0 :
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                elif obj_meta.class_id == 0 :
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)

            except StopIteration:
                break
        
        # folder_path = 'frames'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
            
        # if save_image:
        #     img_path = "{}/frame_{}.jpg".format(folder_path, frame_number)
        #     cv2.imwrite(img_path, frame_copy)
        

        saved_count["stream_{}".format(frame_meta.pad_index)] += 1
        # if forward_count["show_image"] == 1 and forward_count["time_skip"] == 0:
            # print("show_image")

        # print("show_image", forward_count["show_image"])
        # print("time_skip", forward_count["time_skip"])
        # print("stream_index", "stream_{}".format(frame_meta.pad_index))
        # print("saved_count", saved_count)
        


        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK

def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE:0,
        PGIE_CLASS_ID_PERSON:0,
        PGIE_CLASS_ID_BICYCLE:0,
        PGIE_CLASS_ID_ROADSIGN:0
    }
    num_rects=0


    relative_path = '/'.join(stream_path.split('/')[-3:-1]) 
    # folder_path = f'/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/BV Bạch Mai_infered/{relative_path}/imgs'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    folder_name = relative_path
    folder_path = f'/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/BV Bạch Mai_infered/{folder_name}/imgs'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list


    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list

        is_first_obj = True
        save_image = False

        # saved_count["stream_{}".format(frame_meta.pad_index)] = 0
        
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            
            # mask_data
            mask_meta = obj_meta.mask_params

            #position_labels
            array_mask = mask_meta.get_mask_array()
            array_mask = array_mask.reshape(480, 480)    
            array_mask = sigmoid(array_mask)
            threshold = 0.5
            array_mask = (array_mask > threshold).astype(float) * 1
            mask = array_mask >= 1
            indices = np.argwhere(mask)

            intersection = np.logical_and(array_mask, forward_count["bubble_mask"]).sum()
            union = np.logical_or(array_mask, forward_count["bubble_mask"]).sum()
            iou = intersection / union if union > 0 else 0
            


            if (indices.size > 0) and (obj_meta.class_id != 0) and (forward_count["forward_bool"] >= 1) and (saved_count["stream_0"] % int(forward_count["show_rate"])) == 0 and ((iou <= 0.1)) :
                
                #save images
                if is_first_obj:
                    
                    is_first_obj = False
                    # Getting Image data using nvbufsurface
                    # the input should be address of buffer and batch_id
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    
                    # convert python array into numpy array format in the copy mode.
                    frame_copy = np.array(n_frame, copy=True, order='C')

                    # convert the array into cv2 default color format           
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                    frame_copy = cv2.resize(frame_copy, (480, 480))


                    if platform_info.is_integrated_gpu():
                        # If Jetson, since the buffer is mapped to CPU for retrieval, it must also be unmapped 
                        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id) # The unmap call should be made after operations with the original array are complete.
                                                                                            #  The original array cannot be accessed after this call.

                    if forward_count["show_image"] == 1 and forward_count["time_skip"] == 0:
                        img_path = "{}/frame_{}.jpg".format(folder_path, frame_number)
                        cv2.imwrite(img_path, frame_copy)
                        # txt_path = "{}/frame_{}.txt".format(folder_path, frame_number)
                        # np.savetxt(txt_path, frame_copy.flatten(), fmt='%d')
                        # frame_list = frame_copy.tolist()
                        # json_path = "{}/frame_{}.json".format(folder_path, frame_number)
                        # with open(json_path, 'w') as json_file:
                        #     json.dump(frame_list, json_file)
                # save_image = True
                
            else:
                
                min_row, min_col = 0,0

            try: 
                l_obj=l_obj.next

            except StopIteration:
                break
        
        
            
        # if save_image:
            

        saved_count["stream_{}".format(frame_meta.pad_index)] += 1
        # print("stream_index", "stream_{}".format(frame_meta.pad_index))
        # print("saved_count", saved_count)
        


        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK

def main(args):
    input_source = "video"

    global platform_info
    platform_info = PlatformInfo()
    # Standard GStreamer initialization
    Gst.init(None)

    print("platform_info.is_integrated_gpu()", platform_info.is_integrated_gpu())

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    
    if input_source == "video":
        # Source element for reading from the file
        print("Creating Source \n ")
        source = Gst.ElementFactory.make("filesrc", "file-source")
        if not source:
            sys.stderr.write(" Unable to create Source \n")
        source.set_property('location', stream_path)
        
        # Since the data format in the input file is elementary h264 stream,
        # we need a h264parser
        print("Creating H264Parser \n")
        h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
        if not h264parser:
            sys.stderr.write(" Unable to create h264 parser \n")
        
        # Use nvdec_h264 for hardware accelerated decode on GPU
        print("Creating Decoder \n")
        decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
        if not decoder:
            sys.stderr.write(" Unable to create Nvv4l2 Decoder \n")
        
        pipeline.add(source)
        pipeline.add(h264parser)
        pipeline.add(decoder)

        source.link(h264parser)
        h264parser.link(decoder)

        srcpad = decoder.get_static_pad("src")
        if not srcpad:
            sys.stderr.write(" Unable to get source pad of decoder \n")
    else:
        source = Gst.ElementFactory.make("v4l2src", "usb-cam-source")
        if not source:
            sys.stderr.write(" Unable to create Source \n")
        source.set_property('device', '/dev/video0') # device: /dev/video0

        caps_v4l2src = Gst.ElementFactory.make("capsfilter", "v4l2src_caps")
        if not caps_v4l2src:
            sys.stderr.write(" Unable to create v4l2src capsfilter \n")
        caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))

        # videoconvert to make sure a superset of raw formats are supported
        vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
        if not vidconvsrc:
            sys.stderr.write(" Unable to create videoconvert \n")

        # nvvideoconvert to convert incoming raw buffers to NVMM Mem (NvBufSurface API)
        nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
        if not nvvidconvsrc:
            sys.stderr.write(" Unable to create Nvvideoconvert \n")

        caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
        if not caps_vidconvsrc:
            sys.stderr.write(" Unable to create capsfilter \n")
        caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))

        pipeline.add(source)
        pipeline.add(caps_v4l2src)
        pipeline.add(vidconvsrc)
        pipeline.add(nvvidconvsrc)
        pipeline.add(caps_vidconvsrc)

        source.link(caps_v4l2src)
        caps_v4l2src.link(vidconvsrc)
        vidconvsrc.link(nvvidconvsrc)
        nvvidconvsrc.link(caps_vidconvsrc)

        srcpad = caps_vidconvsrc.get_static_pad("src")
        if not srcpad:
            sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    if not sgie:
        sys.stderr.write(" Unable to create sgie \n")

    bubbles_gie = Gst.ElementFactory.make("nvinfer", "tertiary-inference")
    if not bubbles_gie:
        sys.stderr.write(" Unable to create bubbles_gie \n")
    
    
    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    
    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvosd.set_property('display-mask', 1)
    nvosd.set_property('display-text', 1)
    nvosd.set_property('display-bbox', 0)
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")

    if not nvvidconv_postosd:
        sys.stderr.write(" Unable to create nvvidconv_postosd \n")
    
    #Create tee to separate pipeline
    tee = Gst.ElementFactory.make("tee", "nvsink-tee")
    if not tee:
        sys.stderr.write(" Unable to create tee \n")

    queue1 = Gst.ElementFactory.make("queue", "nvtee-que1")
    if not queue1:
        sys.stderr.write(" Unable to create queue1 \n")

    queue2 = Gst.ElementFactory.make("queue", "nvtee-que2")
    if not queue2:
        sys.stderr.write(" Unable to create queue2 \n")

    # queue3 = Gst.ElementFactory.make("queue", "nvtee-que3")
    # if not queue3:
    #     sys.stderr.write(" Unable to create queue1 \n")

    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    if enc_type == 0:
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    else:
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw, format=I420"))
    
    # Make the encoder
    if codec == "H264":
        if enc_type == 0:
            encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        else:
            encoder = Gst.ElementFactory.make("x264enc", "encoder")
        print("Creating H264 Encoder")
    elif codec == "H265":
        if enc_type == 0:
            encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        else:
            encoder = Gst.ElementFactory.make("x265enc", "encoder")
        print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property('bitrate', bitrate)
    if platform_info.is_integrated_gpu() and enc_type == 0:
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        #encoder.set_property('bufapi-version', 1)

    # Make the payload-encode video into RTP packets
    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")
    
    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")
    
    sink.set_property('host', '224.224.255.255')
    sink.set_property('port', updsink_port_num)
    sink.set_property('async', False)
    sink.set_property('sync', 1)
    
    print("Playing file %s " %stream_path)
    
    streammux.set_property('width', 790)
    streammux.set_property('height', 790)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    # streammux.set_property('nvbuf-memory-type', 3)
    # pgie.set_property('config-file-path', "config_infer_primary_esfpnet_single.txt")
    # pgie.set_property('config-file-path', "config_infer_primary_esfpnet_joint.txt")
    pgie.set_property('config-file-path', "/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/config_infer_primary_esfpnet_joint_khoiu.txt")
    sgie.set_property('config-file-path', "/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/config_infer_primary_esfpnet_joint_mgp.txt")
    bubbles_gie.set_property('config-file-path', "/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/config_infer_bubbles_detection.txt")
    
    # mp4
    ##############
    nvvidconv_mp4= Gst.ElementFactory.make("nvvideoconvert", "convertor_mp4")

    if not nvvidconv_mp4:
        sys.stderr.write(" Unable to create nvvidconv_mp4 \n")

    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    if not capsfilter:
        sys.stderr.write(" Unable to create capsfilter \n")
    caps_mp4 = Gst.Caps.from_string("video/x-raw, format=I420") #
    capsfilter.set_property("caps", caps_mp4)
    
    encoder_mp4 = Gst.ElementFactory.make("avenc_mpeg4", "encoder_mp4")
    if not encoder_mp4:
        sys.stderr.write(" Unable to create encoder_mp4 \n")
    encoder_mp4.set_property("bitrate", 2000000)

    codeparser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4-parser")
    if not codeparser:
        sys.stderr.write(" Unable to create code parser \n")
    
    container = Gst.ElementFactory.make("qtmux", "qtmux")
    if not container:
        sys.stderr.write(" Unable to create code parser \n")

    folder_name = '/'.join(stream_path.split('/')[-3:]).replace('.h264', '')
    folder_mp4 = f'/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/BV Bạch Mai_infered/{folder_name}.mp4'
    relative_path = '/'.join(stream_path.split('/')[-3:-1]) 
    relative_mp4 = f'/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/BV Bạch Mai_infered/{relative_path}'
    if not os.path.exists(relative_mp4):
        os.makedirs(relative_mp4)

    file_sink = Gst.ElementFactory.make("filesink", "filesink")
    if not file_sink:
        sys.stderr.write(" Unable to create file sink \n")
    file_sink.set_property("location", folder_mp4)
    file_sink.set_property("sync", 0)
    ##############/opt/nvidia/deepstream/deepstream-7.0/sources/apps/sample_apps/Deepstream/deepstream-esfpnet-rtsp-out/frames/BV Bạch Mai_infered/CaBenh/BU0081


    ##############
    nvvidconv_frames = Gst.ElementFactory.make("nvvideoconvert", "convertor_frames")
    if not nvvidconv_frames:
        sys.stderr.write(" Unable to create nvvidconv_frames \n")

    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")

    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)

    ##############

    if not platform_info.is_integrated_gpu():
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv_frames.set_property("nvbuf-memory-type", mem_type)

    print("Adding elements to Pipeline \n")
    
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(bubbles_gie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(tee)
    pipeline.add(queue1)
    pipeline.add(queue2)

    #mp4
    pipeline.add(nvvidconv_mp4)
    pipeline.add(capsfilter)
    pipeline.add(encoder_mp4)
    pipeline.add(codeparser)
    pipeline.add(container)
    pipeline.add(file_sink)

    #rtsp
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    #frames
    pipeline.add(nvvidconv_frames)
    pipeline.add(filter1)

    # Link the elements together:
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> nvvidconv_postosd -> 
    # caps -> encoder -> rtppay -> udpsink
    
    print("Linking elements in the Pipeline \n")
    
    sinkpad = streammux.request_pad_simple("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    
    
    
    srcpad.link(sinkpad)
    streammux.link(sgie)
    sgie.link(bubbles_gie)
    bubbles_gie.link(pgie)
    pgie.link(nvvidconv)
    # bubbles_gie.link(nvvidconv)
    nvvidconv.link(filter1)
    filter1.link(nvvidconv_frames)
    nvvidconv_frames.link(nvosd)

    # srcpad.link(sinkpad)
    # streammux.link(bubbles_gie)
    # bubbles_gie.link(sgie)
    # sgie.link(pgie)
    # pgie.link(nvvidconv)
    # nvvidconv.link(filter1)
    # filter1.link(nvvidconv_frames)
    # nvvidconv_frames.link(nvosd)


    nvosd.link(tee)

    queue1.link(nvvidconv_mp4)
    nvvidconv_mp4.link(capsfilter)
    capsfilter.link(encoder_mp4)
    encoder_mp4.link(codeparser)
    codeparser.link(container)
    container.link(file_sink)

    queue2.link(nvvidconv_postosd)
    # nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    # queue3.link()

    sink_pad = queue1.get_static_pad("sink")
    tee_mp4 = tee.request_pad_simple('src_%u')
    tee_rtsp = tee.request_pad_simple("src_%u")
    # tee_frames = tee.request_pad_simple("src_%u")
    if not tee_mp4 or not tee_rtsp :
        sys.stderr.write("Unable to get request pads\n")
    tee_mp4.link(sink_pad)
    sink_pad = queue2.get_static_pad("sink")
    tee_rtsp.link(sink_pad)
    # sink_pad = queue3.get_static_pad("sink")
    # tee_frames.link(sink_pad)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    # GLib.unix_signal_add(-100, 2, handle_sigint, None)
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    # GLib.unix_signal_add(-100, 2, handle_sigint(pipeline, bus), None)
    
    bus.connect ("message", bus_call, loop)
    
    # Start streaming
    rtsp_port_num = 8555
    
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)
    
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch( "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (updsink_port_num, codec))
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)
    
    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)
    
    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    # osdsinkpad = nvosd.get_static_pad("sink")
    # if not osdsinkpad:
    #     sys.stderr.write(" Unable to get sink pad of nvo    sd \n")
    
    # osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    pgie_src_pad = pgie.get_static_pad("src")
    if not pgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    sgie_src_pad = sgie.get_static_pad("src")
    if not sgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_src_pad_buffer_probe, 0)

    bubbles_gie_src_pad = bubbles_gie.get_static_pad("src")
    if not bubbles_gie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    bubbles_gie_src_pad.add_probe(Gst.PadProbeType.BUFFER, bubbles_gie_src_pad_buffer_probe, 0)
    
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)

def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--input",
                  help="Path to input H264 elementry stream", required=False)
    parser.add_argument("-c", "--codec", default="H264",
                  help="RTSP Streaming Codec H264/H265 , default=H264", choices=['H264','H265'])
    parser.add_argument("-b", "--bitrate", default=4000000,
                  help="Set the encoding bitrate ", type=int)
    parser.add_argument("-e", "--enc_type", default=0,
                  help="0:Hardware encoder , 1:Software encoder , default=0", choices=[0, 1], type=int)
                  
    # Check input arguments
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    global codec
    global bitrate
    global stream_path
    global enc_type
    codec = args.codec
    bitrate = args.bitrate
    stream_path = args.input
    enc_type = args.enc_type
    return 0

if __name__ == '__main__':
    parse_args()
    sys.exit(main(sys.argv))

