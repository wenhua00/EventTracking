# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Code sample showing how to use Metavision SDK to display results of dense optical flow.
"""

import numpy as np
import os
import h5py
import math
import cv2
import sys
import time
import rosbag  # 新增
import json   # 新增
from datetime import datetime  # 新增
sys.path.append("/usr/lib/python3/dist-packages/")  # Required to import metavision_core

from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
from metavision_sdk_cv import PlaneFittingFlowAlgorithm, TimeGradientFlowAlgorithm, TripletMatchingFlowAlgorithm, \
    DenseFlowFrameGeneratorAlgorithm, SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent
import rospy
from std_msgs.msg import String
from skvideo.io import FFmpegWriter
from enum import Enum
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


bias_diff_off = -0
bias_diff_on = 0
bias_hpf = 0
bias_fo = 0
bias_refr = 0
cnt_evs = 0.0
early_warning_time = 0.0400
last_frame_time = 0

track_list = [] #x y w h nowtime evsnumber start_time track_id
cnt_track_id = 0

class FlowType(Enum):
    PlaneFitting = "PlaneFitting"
    TimeGradient = "TimeGradient"
    TripletMatching = "TripletMatching"

    def __str__(self):
        return self.value


# 新增全局变量用于时间戳对齐和数据存储
aligned_data_storage = []
bag_reader = None
bag_writer = None
time_offset = 0.0
start_recording_time = None
should_start_saving = False  # 新增：控制开始保存的标志

# 新增函数：初始化bag文件读取
def init_bag_reader(bag_path):
    """初始化bag文件读取器"""
    global bag_reader, bag_data_cache, time_offset, start_recording_time
    
    try:
        bag_reader = rosbag.Bag(bag_path, 'r')
        print(f"Successfully opened bag file: {bag_path}")
        
        # 读取bag文件的起始时间
        bag_start_time = None
        for topic, msg, t in bag_reader.read_messages():
            bag_start_time = t.to_sec()
            break
        
        if bag_start_time:
            start_recording_time = time.time()
            time_offset = bag_start_time - start_recording_time
            print(f"Bag start time: {bag_start_time}, Time offset: {time_offset}")
        
        return True
    except Exception as e:
        print(f"Error opening bag file: {e}")
        return False

# 新增函数：从bag文件中获取对应时间戳的数据
def get_bag_data_at_timestamp(target_timestamp, tolerance=0.05):
    """根据时间戳从bag文件中获取数据"""
    global bag_reader, bag_data_cache
    
    if not bag_reader:
        return None
    
    # 检查缓存
    cache_key = round(target_timestamp, 3)
    if cache_key in bag_data_cache:
        return bag_data_cache[cache_key]
    
    # 在bag文件中查找最接近的时间戳数据
    closest_data = None
    min_time_diff = float('inf')
    
    try:
        for topic, msg, t in bag_reader.read_messages():
            bag_timestamp = t.to_sec()
            time_diff = abs(bag_timestamp - target_timestamp)
            
            if time_diff < min_time_diff and time_diff <= tolerance:
                min_time_diff = time_diff
                closest_data = {
                    'timestamp': bag_timestamp,
                    'topic': topic,
                    'message': msg,
                    'time_diff': time_diff
                }
        
        # 缓存结果
        if closest_data:
            bag_data_cache[cache_key] = closest_data
        
        return closest_data
    except Exception as e:
        print(f"Error reading bag data: {e}")
        return None

# 新增函数：保存对齐后的数据
def save_aligned_data(time_surface, bbox_info, bag_data, timestamp, frame_id):
    """保存对齐后的数据"""
    global aligned_data_storage
    
    aligned_item = {
        'timestamp': timestamp,
        'frame_id': frame_id,
        'time_surface': time_surface.copy(),
        'bbox_info': bbox_info,
        'bag_data': bag_data,
        'save_time': datetime.now().isoformat()
    }
    
    aligned_data_storage.append(aligned_item)

# 新增函数：导出合并后的数据
def export_merged_data(output_path="merged_data.bag"):
    """导出合并后的数据到新的bag文件"""
    global aligned_data_storage
    
    if not aligned_data_storage:
        print("No aligned data to export")
        return
    
    try:
        bridge = CvBridge()
        
        with rosbag.Bag(output_path, 'w') as outbag:
            for item in aligned_data_storage:
                ros_time = rospy.Time.from_sec(item['timestamp'])
                
                # 保存time_surface
                time_surface_msg = bridge.cv2_to_imgmsg(item['time_surface'], encoding="bgr8")
                time_surface_msg.header.stamp = ros_time
                time_surface_msg.header.frame_id = f"frame_{item['frame_id']}"
                outbag.write('/cd_output_img_topic', time_surface_msg, ros_time)
                
                # 保存bbox信息
                bbox_msg = String()
                bbox_msg.data = json.dumps(item['bbox_info'])
                outbag.write('/event_camera/bbox_data', bbox_msg, ros_time)
                
                # 保存原始bag数据（如果有的话）
                if item['bag_data']:
                    outbag.write(item['bag_data']['topic'], item['bag_data']['message'], ros_time)
        
        print(f"Merged data exported to: {output_path}")
        
        # 保存统计报告
        report_path = output_path.replace('.bag', '_report.json')
        report_data = {
            'total_frames': len(aligned_data_storage),
            'time_range': {
                'start': min(item['timestamp'] for item in aligned_data_storage),
                'end': max(item['timestamp'] for item in aligned_data_storage)
            },
            'bbox_statistics': {
                'total_detections': sum(len(item['bbox_info']) for item in aligned_data_storage),
                'frames_with_detection': sum(1 for item in aligned_data_storage if len(item['bbox_info']) > 0)
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error exporting merged data: {e}")

# 修改全局变量
aligned_data_storage = []
bag_reader = None
bag_writer = None
time_offset = 0.0
start_recording_time = None
should_start_saving = False  # 新增：控制开始保存的标志

# 修改：根据文件名决定开始保存的时机
def determine_start_time_from_filename(raw_filename, bag_path):
    """根据RAW文件名确定开始保存的时机"""
    global should_start_saving, bag_writer, time_offset, start_recording_time
    
    try:
        # 从RAW文件名中提取时间戳信息
        # 假设文件名格式为: recording_1753612683.523199.raw
        if "recording_" in raw_filename:
            timestamp_str = raw_filename.replace("recording_", "").replace(".raw", "")
            raw_timestamp = float(timestamp_str)
            start_recording_time = raw_timestamp
            print(f"Event alignment will start from .raw file timestamp: {start_recording_time}")

            print(f"RAW file timestamp: {raw_timestamp}")
            
            # 生成输出文件路径
            output_bag_path = bag_path.replace('.bag', '_with_events.bag')
            
            # 打开新的bag文件进行写入
            bag_writer = rosbag.Bag(output_bag_path, 'w')
            
            # 读取原始bag文件并复制所有消息，同时获取时间戳信息
            original_bag = rosbag.Bag(bag_path, 'r')
            
            print(f"Copying original bag messages to: {output_bag_path}")
            message_count = 0
            first_message_time = None
            imu_start_time = None
            
            for topic, msg, t in original_bag.read_messages():
                if first_message_time is None:
                    first_message_time = t.to_sec()
                    print(f"First bag message at: {first_message_time}")
                
                # # 记录第一个IMU消息的时间戳作为参考基准
                # if topic == '/livox/imu' and imu_start_time is None:
                #     imu_start_time = t.to_sec()
                #     print(f"First IMU message at: {imu_start_time}")
                    
                #     # 修改：以IMU开始时间为基准，计算偏移量
                #     # 让事件数据的第一帧对应IMU的开始时间
                #     time_offset = imu_start_time  # 直接使用IMU开始时间作为偏移基准
                #     start_recording_time = imu_start_time  # 记录IMU开始时间
                #     should_start_saving = True
                    
                #     print(f"Time offset set to IMU start time: {time_offset}")
                #     print(f"Event data will be aligned to IMU timeline starting from: {imu_start_time}")
                should_start_saving = True
                # 原封不动地复制原始消息
                bag_writer.write(topic, msg, t)
                message_count += 1
            
            original_bag.close()
            print(f"Copied {message_count} original messages")
            print(f"Will add new topics with aligned timestamps:")
            print(f"  - /event_camera/time_surface (sensor_msgs/Image)")
            print(f"  - /event_camera/bbox_data (std_msgs/String)")
            return True
            
    except Exception as e:
        print(f"Error parsing filename or copying bag: {e}")
        return False

# 修改保存函数，使用基于IMU开始时间的时间戳
def save_event_data_to_bag(time_surface, bbox_info, timestamp, frame_id):
    """保存事件数据到bag文件，使用与IMU对齐的时间戳"""
    global bag_writer, time_offset, start_recording_time, initial_event_time
    
    if not bag_writer or not should_start_saving:
        return
    
    try:
        bridge = CvBridge()
        
        # 修改：计算相对于事件开始的时间差，然后加上IMU开始时间
        # 第一次调用时记录初始事件时间
        global initial_event_time
        if 'initial_event_time' not in globals() or initial_event_time is None:
            initial_event_time = timestamp
            print(f"Initial event time recorded (relative zero): {initial_event_time}")

        relative_time = timestamp - initial_event_time
        aligned_timestamp = start_recording_time + relative_time

        ros_time = rospy.Time.from_sec(aligned_timestamp)
        
        # 保存time_surface图像
        time_surface_msg = bridge.cv2_to_imgmsg(time_surface, encoding="bgr8")
        time_surface_msg.header.stamp = ros_time
        time_surface_msg.header.frame_id = f"event_frame_{frame_id}"
        bag_writer.write('/cd_output_img_topic', time_surface_msg, ros_time)
        
        if bbox_info:
            bbox_strings = []
            for box in bbox_info:
                # 从字典中提取所需字段
                bbox_x = box.get('x', 0)
                bbox_y = box.get('y', 0)
                bbox_width = box.get('width', 0)
                bbox_heigth = box.get('height', 0)
                filtered_points_number = box.get('point_count', 0)
                mean_vx = box.get('mean_vx', 0.0)
                mean_vy = box.get('mean_vy', 0.0)
                
                # 格式化为字符串
                bbox_str = f"{bbox_x},{bbox_y},{bbox_width},{bbox_heigth},{filtered_points_number},{mean_vx:.2f},{mean_vy:.2f}"
                bbox_strings.append(bbox_str)
            
            # 将所有bbox字符串合并为一个消息
            final_bbox_data = "\n".join(bbox_strings)
            
            bbox_msg = String()
            bbox_msg.data = final_bbox_data
            bag_writer.write('/bbox', bbox_msg, ros_time)
        
        return aligned_timestamp  # 返回对齐后的时间戳用于显示
        
    except Exception as e:
        print(f"Error saving to bag: {e}")
        return None

# 全局变量，仅用于写入合并后的bag文件
bag_writer = None

def prepare_merged_bag(bag_path):
    """
    准备用于合并的bag文件。
    这会打开原始bag，将其所有内容复制到一个新的bag文件，并保持新文件打开以供写入。
    """
    global bag_writer
    try:
        # 生成输出文件路径
        output_bag_path = bag_path.replace('.bag', '_with_events.bag')
        # 打开新的bag文件进行写入
        bag_writer = rosbag.Bag(output_bag_path, 'w')
        
        print(f"正在从 {bag_path} 复制原始bag消息到 {output_bag_path}")
        # 打开原始bag文件进行读取
        with rosbag.Bag(bag_path, 'r') as original_bag:
            message_count = 0
            # 复制所有消息
            for topic, msg, t in original_bag.read_messages():
                bag_writer.write(topic, msg, t)
                message_count += 1
        print(f"已复制 {message_count} 条原始消息。现在可以添加事件数据。")
        return True
    except Exception as e:
        print(f"准备合并的bag文件时出错: {e}")
        if bag_writer:
            bag_writer.close()
        return False

# def save_event_data_to_bag(time_surface, bbox_info, final_timestamp, frame_id):
#     """
#     使用预先计算好的最终时间戳，将事件数据（图像和BBox）保存到已打开的bag文件中。
#     """
#     global bag_writer
#     if not bag_writer:
#         return
    
#     try:
#         bridge = CvBridge()
#         ros_time = rospy.Time.from_sec(final_timestamp)
        
#         # 1. 保存time_surface图像
#         time_surface_msg = bridge.cv2_to_imgmsg(time_surface, encoding="bgr8")
#         time_surface_msg.header.stamp = ros_time
#         time_surface_msg.header.frame_id = f"event_frame_{frame_id}"
#         bag_writer.write('/event_camera/time_surface', time_surface_msg, ros_time)
        
#         # 2. 保存bbox信息
#         bbox_msg = String()
#         bbox_data = {
#             'frame_id': frame_id,
#             'timestamp': final_timestamp, # 使用最终计算出的时间戳
#             'bboxes': bbox_info,
#             'bbox_count': len(bbox_info)
#         }
#         bbox_msg.data = json.dumps(bbox_data)
#         bag_writer.write('/event_camera/bbox_data', bbox_msg, ros_time)
        
#     except Exception as e:
#         print(f"保存事件数据到bag时出错: {e}")

def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Dense Optical Flow sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group(
        "Input", "Arguments related to input sequence.")
    input_group.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.33333")
    # 修改bag文件参数说明
    input_group.add_argument(
        '--bag-file', dest='bag_file_path', default="",
        help="Path to input bag file to merge with event data")
    input_group.add_argument(
        '--output-merged-bag', dest='output_merged_bag', default="",
        help="Path to output merged bag file (auto-generated if not specified)")
    
    # 移除时间容差参数，不再需要
    input_group.add_argument(
        '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")
    input_group.add_argument(
        '--dt-step', type=int, default=20000, dest="dt_step",
        help="Time processing step (in us), used as iteration delta_t period, visualization framerate and accumulation time.")

    algo_settings_group = parser.add_argument_group(
        "Algo Settings", "Arguments related to algorithmic configuration.")
    algo_settings_group.add_argument(
        "--flow-type", dest="flow_type", type=FlowType, choices=list(FlowType),
        default=FlowType.TripletMatching, help="Chosen type of dense flow algorithm to run")
    algo_settings_group.add_argument(
        "-r", "--receptive-field-radius", dest="receptive_field_radius", type=float, default=3,  #default 3
        help="Radius of the receptive field used for flow estimation, in pixels.")
    algo_settings_group.add_argument("--min-flow", dest="min_flow_mag", type=float,  default=1500.,  #default 1000
                                     help="Minimum observable flow magnitude, in px/s.")
    algo_settings_group.add_argument("--max-flow", dest="max_flow_mag", type=float,  default=5000.,
                                     help="Maximum observable flow magnitude, in px/s.")
    algo_settings_group.add_argument("--stc-filter-thr", dest="stc_filter_thr", type=int,  default=5000,
                                     help="Length of the time window for filtering (in us).")
    algo_settings_group.add_argument(
        "--visu-scale", dest="visualization_flow_scale", type=float, default=0.8,
        help="Flow magnitude used to scale the upper bound of the flow visualization, in px/s. If negative, will use 1/5-th of maximum flow magnitude.")

    output_flow_group = parser.add_argument_group(
        "Output flow", "Arguments related to output optical flow.")
    output_flow_group.add_argument(
        "--output-sparse-npy-filename", dest="output_sparse_npy_filename",
        help="If provided, the predictions will be saved as numpy structured array of EventOpticalFlow. In this "
        "format, the flow vx and vy are expressed in pixels per second.")
    output_flow_group.add_argument(
        "--output-dense-h5-filename", dest="output_dense_h5_filename",
        help="If provided, the predictions will be saved as a sequence of dense flow in HDF5 data. The flows are "
        "averaged pixelwise over timeslices of --dt-step. The dense flow is expressed in terms of "
        "pixels per timeslice (of duration dt-step), not in pixels per second.")
    output_flow_group.add_argument(
        '-o', '--out-video', dest='out_video', type=str, default="",
        help="Path to an output AVI file to save the resulting video.")
    output_flow_group.add_argument(
        '--fps', dest='fps', type=int, default=30,
        help="replay fps of output video")
    output_flow_group.add_argument(
        '--no-legend', dest='show_legend', action='store_false',
        help="Set to remove the legend from the display")

    args = parser.parse_args()

    if args.output_sparse_npy_filename:
        assert not os.path.exists(args.output_sparse_npy_filename)
    if args.output_dense_h5_filename:
        assert not os.path.exists(args.output_dense_h5_filename)
    if args.visualization_flow_scale <= 0:
        args.visualization_flow_scale = 1

    return args


def accumulate_estimated_flow_data(
        args, width, height, processing_ts, flow_buffer, all_flow_events, all_dense_flows_start_ts,
        all_dense_flows_end_ts, all_dense_flows):
    """Accumulates estimated flow data in buffers to dump estimation results"""
    if args.output_sparse_npy_filename:
        all_flow_events.append(flow_buffer.numpy().copy())
    if args.output_dense_h5_filename:
        all_dense_flows_start_ts.append(
            processing_ts - args.dt_step)
        all_dense_flows_end_ts.append(processing_ts)
        flow_np = flow_buffer.numpy()
        if flow_np.size == 0:
            all_dense_flows.append(
                np.zeros((2, height, width), dtype=np.float32))
        else:
            xs, ys, vx, vy = flow_np["x"], flow_np["y"], flow_np["vx"], flow_np["vy"]
            coords = np.stack((ys, xs))
            abs_coords = np.ravel_multi_index(coords, (height, width))
            counts = np.bincount(abs_coords, weights=np.ones(flow_np.size),
                                 minlength=height*width).reshape(height, width)
            flow_x = np.bincount(
                abs_coords, weights=vx, minlength=height*width).reshape(height, width)
            flow_y = np.bincount(
                abs_coords, weights=vy, minlength=height*width).reshape(height, width)
            mask_multiple_events = counts > 1
            flow_x[mask_multiple_events] /= counts[mask_multiple_events]
            flow_y[mask_multiple_events] /= counts[mask_multiple_events]

            # flow expressed in pixels per delta_t
            flow_x *= args.dt_step * 1e-6
            flow_y *= args.dt_step * 1e-6
            flow = np.stack((flow_x, flow_y)).astype(np.float32)
            all_dense_flows.append(flow)



def dump_estimated_flow_data(
        args, width, height, all_flow_events, all_dense_flows_start_ts, all_dense_flows_end_ts, all_dense_flows):
    """Write accumulated flow results to output files"""
    try:
        if args.output_sparse_npy_filename:
            print("Writing output file: ", args.output_sparse_npy_filename)
            all_flow_events = np.concatenate(all_flow_events)
            np.save(args.output_sparse_npy_filename, all_flow_events)
        if args.output_dense_h5_filename:
            print("Writing output file: ", args.output_dense_h5_filename)
            flow_start_ts = np.array(all_dense_flows_start_ts)
            flow_end_ts = np.array(all_dense_flows_end_ts)
            flows = np.stack(all_dense_flows)
            N = flow_start_ts.size
            assert flow_end_ts.size == N
            assert flows.shape == (N, 2, height, width)
            dirname = os.path.dirname(args.output_dense_h5_filename)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            flow_h5 = h5py.File(args.output_dense_h5_filename, "w")
            flow_h5.create_dataset(
                "flow_start_ts", data=flow_start_ts, compression="gzip")
            flow_h5.create_dataset(
                "flow_end_ts", data=flow_end_ts, compression="gzip")
            flow_h5.create_dataset("flow", data=flows.astype(
                np.float32), compression="gzip")
            flow_h5["flow"].attrs["input_file_name"] = os.path.basename(
                args.event_file_path)
            flow_h5["flow"].attrs["checkpoint_path"] = "metavision_dense_optical_flow"
            flow_h5["flow"].attrs["event_input_height"] = height
            flow_h5["flow"].attrs["event_input_width"] = width
            flow_h5["flow"].attrs["delta_t"] = args.dt_step
            flow_h5.close()
    except Exception as e:
        print(e)
        raise

def calculate_diou(bbox1, bbox2):
    # 提取第一个边界框的坐标
    x1_min, y1_min = bbox1[0], bbox1[1]
    x1_max, y1_max = x1_min + bbox1[2], y1_min + bbox1[3]   
    # 提取第二个边界框的坐标
    x2_min, y2_min = bbox2[0], bbox2[1]
    x2_max, y2_max = x2_min + bbox2[2], y2_min + bbox2[3]  
    # 计算交集区域的坐标
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)  
    # 交集的宽度和高度
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min) 
    # 交集区域面积
    inter_area = inter_width * inter_height 
    # 第一个边界框的面积
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)  
    # 第二个边界框的面积
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min) 
    # 并集面积 = 两个面积之和 - 交集面积
    union_area = bbox1_area + bbox2_area - inter_area
    # IoU 计算
    iou = inter_area / union_area if union_area > 0 else 0
    # 计算边界框的中心点
    bbox1_center = ((x1_min + x1_max) / 2, (y1_min + y1_max) / 2)
    bbox2_center = ((x2_min + x2_max) / 2, (y2_min + y2_max) / 2)
    # 计算中心点之间的欧几里得距离
    rho = np.sqrt((bbox1_center[0] - bbox2_center[0]) ** 2 + (bbox1_center[1] - bbox2_center[1]) ** 2)
    # 计算包围两个边界框的最小矩形的对角线长度
    enclosing_x_min = min(x1_min, x2_min)
    enclosing_y_min = min(y1_min, y2_min)
    enclosing_x_max = max(x1_max, x2_max)
    enclosing_y_max = max(y1_max, y2_max)
    c = np.sqrt((enclosing_x_max - enclosing_x_min) ** 2 + (enclosing_y_max - enclosing_y_min) ** 2)
    # DIoU 计算
    diou = iou - (rho ** 2) / (c ** 2)
    return diou
    

def track_(bbox):
    global cnt_track_id,track_list
    track_id = None
    max_diou = -1
    if(len(track_list) == 0):
        cnt_track_id = cnt_track_id + 1
        track_list.append(bbox + [bbox[4],cnt_track_id])
        return track_list[-1],False
    else:
        for i in range(len(track_list)):
            diou = calculate_diou(track_list[i][:-1], bbox)
            if(diou > -0.2 and diou > max_diou):
                max_diou = diou
                track_id = i
        if(track_id is not None):
            track_list[track_id][:6] = bbox
            flag = False
            if(track_list[track_id][4] - track_list[track_id][-2] >= early_warning_time):
                flag = True
            return track_list[track_id],flag
        else:
            cnt_track_id = cnt_track_id + 1
            track_list.append(bbox + [bbox[4],cnt_track_id])
            return track_list[-1],False

# def filter_foreground_events(events, min_count=2):
#     # events: 结构化数组，含x,y
#     coords = np.stack([events['y'], events['x']], axis=1)
#     unique, counts = np.unique(coords, axis=0, return_counts=True)
#     mask = np.isin(coords.tolist(), unique[counts > min_count].tolist())
#     return events[mask]

def main():
    """ Main """
    args = parse_args()
    
    # 在函数开头声明所有全局变量
    global initial_event_time, track_list, last_frame_time
    
    # 修改：根据文件名决定保存时机
    if args.bag_file_path:
        raw_filename = os.path.basename(args.event_file_path)
        
        if not determine_start_time_from_filename(raw_filename, args.bag_file_path):
            print("Failed to determine start time from filename, continuing without bag integration")

    # ROS node initialization
    rospy.init_node('bbox_publisher', anonymous=True)
    
    # 修改：创建发布器但不立即发布，等待时间对齐
    bbox_pub = rospy.Publisher('/bbox/'+args.event_file_path[-5:], String, queue_size=10)
    filtered_points_pub = rospy.Publisher('/filtered_points/'+args.event_file_path[-5:], String, queue_size=10)
    bridge = CvBridge()
    cd_output_img_pub = rospy.Publisher('/cd_output_img_topic/'+args.event_file_path[-5:], Image, queue_size=3)

    device = initiate_device(path=args.event_file_path)

    if device.get_i_ll_biases():
        device.get_i_ll_biases().set("bias_diff_off", bias_diff_off)
        device.get_i_ll_biases().set("bias_diff_on", bias_diff_on)
        device.get_i_ll_biases().set("bias_hpf", bias_hpf)
        device.get_i_ll_biases().set("bias_fo", bias_fo)
        device.get_i_ll_biases().set("bias_refr", bias_refr)

    mv_iterator = EventsIterator.from_device(device=device,delta_t=args.dt_step)

    # Set ERC to 20Mev/s
    if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device and mv_iterator.reader.device.get_i_erc_module():
        erc_module = mv_iterator.reader.device.get_i_erc_module()
        erc_module.set_cd_event_rate(20000000)
        erc_module.enable(False)

    if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(
            mv_iterator, replay_factor=args.replay_factor)
    height, width = mv_iterator.get_size()

    # Event Frame Generator
    event_frame_gen = OnDemandFrameGenerationAlgorithm(
        width, height, args.dt_step)

    # Dense Optical Flow Algorithm
    if args.flow_type == FlowType.PlaneFitting:
        radius = math.floor(args.receptive_field_radius)
        print(f"Instantiating PlaneFittingFlowAlgorithm with radius={radius}")
        flow_algo_planefitting = PlaneFittingFlowAlgorithm(width, height, radius, -1)
        flow_algo_timegradient = None
        flow_algo_tripletmatching = None
    elif args.flow_type == FlowType.TimeGradient:
        radius = int(args.receptive_field_radius)
        print(f"Instantiating TimeGradientFlowAlgorithm with radius={radius}, min_flow={args.min_flow_mag}, bit_cut=2")
        flow_algo_timegradient = TimeGradientFlowAlgorithm(width, height, radius, args.min_flow_mag, 2)
        flow_algo_planefitting = None
        flow_algo_tripletmatching = None
    else:
        radius = 0.5*args.receptive_field_radius
        print(f"Instantiating TripletMatchingFlowAlgorithm with radius={radius}, min_flow={args.min_flow_mag}, max_flow={args.max_flow_mag}")
        flow_algo_tripletmatching = TripletMatchingFlowAlgorithm(width, height, radius, args.min_flow_mag, args.max_flow_mag)
        flow_algo_planefitting = None
        flow_algo_timegradient = None
    flow_buffer = TripletMatchingFlowAlgorithm.get_empty_output_buffer()

    # Dense Flow Frame Generator
    flow_frame_gen = DenseFlowFrameGeneratorAlgorithm(
        width, height, args.max_flow_mag, args.visualization_flow_scale,
        DenseFlowFrameGeneratorAlgorithm.VisualizationMethod.Arrows,
        DenseFlowFrameGeneratorAlgorithm.AccumulationPolicy.Average)

    # STC filter
    print(f"Instantiating SpatioTemporalContrastAlgorithm with thresh={args.stc_filter_thr}")
    stc_filter = SpatioTemporalContrastAlgorithm(width, height, args.stc_filter_thr, True)
    events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()

    all_flow_events = []
    all_dense_flows = []
    all_dense_flows_start_ts = []
    all_dense_flows_end_ts = []

    # Window - Graphical User Interface
    with Window(title="Fast Moving Obstacle Detection", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        if args.out_video:
            video_name = args.out_video + ".avi"
            writer = FFmpegWriter(video_name, inputdict={'-r': str(args.fps)}, outputdict={
                '-vcodec': 'libx264',
                '-pix_fmt': 'yuv420p',
                '-r': str(args.fps)
            })

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        cd_output_img = np.zeros((height, width, 3), np.uint8)
        flow_output_img = np.zeros((height, width, 3), np.uint8)
        combined_output_img = np.zeros((height, width, 3), np.uint8)
        processing_ts = mv_iterator.start_ts

        if args.flow_type == FlowType.PlaneFitting:
            flow_processor = flow_algo_planefitting
        elif args.flow_type == FlowType.TimeGradient:
            flow_processor = flow_algo_timegradient
        else:
            flow_processor = flow_algo_tripletmatching

        # Process events
        last_time = 0.0
        FPS = 0
        cnt_fps = 0
        cnt_time = 0.0
        skip_frame = 0
        frame_count = 0
        saved_frames_count = 0
        
        for evs in mv_iterator:
            processing_ts += mv_iterator.delta_t
            start_frame_time = time.time()
            EventLoop.poll_and_dispatch()

            # Filter Events using STC
            # stc_filter.process_events(evs, events_buf)
            events_buf=evs
            evs_number = len(evs)
            if evs_number == 0:
                continue

            # if skip_frame:
            #     skip_frame = skip_frame - 1
            #     continue

            # Generate Frame of Events
            event_frame_gen.process_events(events_buf)
            event_frame_gen.generate(processing_ts, cd_output_img)

            # Estimate the flow events
            flow_processor.process_events(events_buf, flow_buffer)
            accumulate_estimated_flow_data(
                args, width, height, processing_ts, flow_buffer, all_flow_events, all_dense_flows_start_ts,
                all_dense_flows_end_ts, all_dense_flows)
                
            # Draw the flow events on top of the events
            flow_frame_gen.process_events(flow_buffer)
            flow_frame_gen.generate(flow_output_img)
            
            curr_time = start_frame_time
            
            cnt_time = cnt_time + curr_time - last_time
            cnt_fps = cnt_fps + 1
            if cnt_time >= 1:
                FPS = cnt_fps
                cnt_time = 0
                cnt_fps = 0
            last_time = curr_time
            
            # Update the display
            cv2.addWeighted(cd_output_img, 0.6, flow_output_img, 0.4, 0, combined_output_img)

            # 收集bbox信息
            current_bbox_info = []

            if len(flow_buffer.numpy()) > 0:
                xyt_selected = np.column_stack((flow_buffer.numpy()['x'][:], flow_buffer.numpy()['y'][:], flow_buffer.numpy()['t'][:]))
                xy_selected = np.column_stack((flow_buffer.numpy()['x'][:], flow_buffer.numpy()['y'][:]))
                
                y_coords = xy_selected[:, 1].astype(int)
                x_coords = xy_selected[:, 0].astype(int)

                non_black_mask = ~np.all(flow_output_img[y_coords, x_coords] == [0, 0, 0], axis=1)

                filtered_points = xy_selected[non_black_mask]
                filtered_points_with_time = xyt_selected[non_black_mask]

                combined_output_img[filtered_points[:, 1].astype(int), filtered_points[:, 0].astype(int)] = [0, 255, 255]

                filtered_points = np.array(filtered_points)

                if filtered_points.size > 5:
                    Q1 = np.percentile(filtered_points, 25, axis=0)
                    Q3 = np.percentile(filtered_points, 75, axis=0)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    filtered_points = filtered_points[
                        (filtered_points[:, 0] >= lower_bound[0]) & (filtered_points[:, 0] <= upper_bound[0]) &
                        (filtered_points[:, 1] >= lower_bound[1]) & (filtered_points[:, 1] <= upper_bound[1])
                    ]

                    filtered_points_number = filtered_points.size
                    if filtered_points_number > 0:
                        min_x, min_y = filtered_points.min(axis=0)
                        max_x, max_y = filtered_points.max(axis=0)
                        bbox_x = (max_x + min_x)//2
                        bbox_y = (max_y + min_y)//2
                        bbox_width = max_x - min_x
                        bbox_heigth = max_y - min_y
                        
                        # 修改后的代码（添加边界检查）
                        if bbox_width > 0 and bbox_heigth > 0:
                            wh_ratio = bbox_width/bbox_heigth if bbox_width>bbox_heigth else bbox_heigth/bbox_width
                            if(wh_ratio<2.5 and min(bbox_width,bbox_heigth)<400 and min(bbox_width,bbox_heigth)>20 and max(bbox_width,bbox_heigth)<400):
                                bbox = [bbox_x,bbox_y,bbox_width,bbox_heigth,curr_time,filtered_points_number]
                                track_bbox,flag = track_(bbox)
                                if(flag):
                                    cv2.rectangle(combined_output_img, (min_x, min_y), (max_x, max_y), color=(0, 0, 255), thickness=6)
                                else:
                                    # cv2.rectangle(combined_output_img, (min_x, min_y), (max_x, max_y), color=(0, 255, 0), thickness=5)
                                    pass
                                
                                # 计算bbox内所有filtered_points的平均vx, vy
                                if filtered_points.shape[0] > 2:
                                    # 找到filtered_points在flow_buffer中的索引
                                    all_flow = flow_buffer.numpy()
                                    # 构建(x, y)元组集合用于快速查找
                                    filtered_set = set((int(x), int(y)) for x, y in filtered_points)
                                    vx_list = []
                                    vy_list = []
                                    for i in range(all_flow.shape[0]):
                                        x, y = int(all_flow['x'][i]), int(all_flow['y'][i])
                                        if (x, y) in filtered_set:
                                            vx_list.append(all_flow['vx'][i])
                                            vy_list.append(all_flow['vy'][i])
                                    if vx_list and vy_list:
                                        mean_vx = np.mean(vx_list)
                                        mean_vy = np.mean(vy_list)
                                    else:
                                        mean_vx = 0.0
                                        mean_vy = 0.0
                                else:
                                    mean_vx = 0.0
                                    mean_vy = 0.0

                                # 保存bbox信息
                                bbox_info = {
                                    'x': int(bbox_x),
                                    'y': int(bbox_y),
                                    'width': int(bbox_width),
                                    'height': int(bbox_heigth),
                                    'min_x': int(min_x),
                                    'min_y': int(min_y),
                                    'max_x': int(max_x),
                                    'max_y': int(max_y),
                                    'point_count': int(filtered_points_number),
                                    'mean_vx': float(mean_vx),
                                    'mean_vy': float(mean_vy),
                                    'track_id': int(track_bbox[-1]),
                                    'is_warning': bool(flag)
                                }
                                current_bbox_info.append(bbox_info)

                                # 修改：使用对齐的时间戳发布消息，而不是当前时间
                                if args.bag_file_path and should_start_saving:
                                    # 计算对齐的时间戳
                                    if 'initial_event_time' not in globals() or initial_event_time is None:
                                        initial_event_time = curr_time
                                    
                                    relative_time = curr_time - initial_event_time
                                    aligned_timestamp = start_recording_time + relative_time
                                    aligned_ros_time = rospy.Time.from_sec(aligned_timestamp)
                                    
                                    # 发布bbox和平均vx,vy，使用对齐的时间戳
                                    bbox_msg = f"{bbox_x},{bbox_y},{bbox_width},{bbox_heigth},{filtered_points_number},{mean_vx:.2f},{mean_vy:.2f}"
                                    bbox_pub.publish(bbox_msg)
                                    
                                    points_msg = '\n'.join([f"{int(x)},{int(y)},{int(t)}" for x, y, t in filtered_points_with_time])
                                    filtered_points_pub.publish(points_msg)
                                    
                                    # 发布图像消息，使用对齐的时间戳
                                    cd_output_img_msg = bridge.cv2_to_imgmsg(cd_output_img, encoding="bgr8")
                                    cd_output_img_msg.header.stamp = aligned_ros_time  # 使用对齐的时间戳
                                    cd_output_img_msg.header.frame_id = "image_frame"
                                    cd_output_img_pub.publish(cd_output_img_msg)
                                else:
                                    # 如果没有bag文件，使用当前时间（原有逻辑）
                                    bbox_msg = f"{bbox_x},{bbox_y},{bbox_width},{bbox_heigth},{filtered_points_number},{mean_vx:.2f},{mean_vy:.2f}"
                                    bbox_pub.publish(bbox_msg)
                                    
                                    points_msg = '\n'.join([f"{int(x)},{int(y)},{int(t)}" for x, y, t in filtered_points_with_time])
                                    filtered_points_pub.publish(points_msg)
                                    
                                    # 发布图像消息
                                    cd_output_img_msg = bridge.cv2_to_imgmsg(cd_output_img, encoding="bgr8")
                                    cd_output_img_msg.header.stamp = rospy.Time.now()  # 使用当前时间
                                    cd_output_img_msg.header.frame_id = "image_frame"
                                    cd_output_img_pub.publish(cd_output_img_msg)
                                
                                cv2.putText(combined_output_img, str(track_bbox[-1]), (min_x, min_y-10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
                        else:
                            # 处理无效边界框（跳过或标记）
                            continue  # 或 wh_ratio = float('inf') 根据业务逻辑调整

            # 修改：使用对齐的时间戳保存到bag文件
            aligned_timestamp = None
            if args.bag_file_path and should_start_saving:
                aligned_timestamp = save_event_data_to_bag(
                    time_surface=cd_output_img,
                    bbox_info=current_bbox_info,
                    timestamp=curr_time,
                    frame_id=frame_count
                )
                
                if aligned_timestamp:
                    saved_frames_count += 1
                    
                    # 显示对齐状态
                    cv2.putText(combined_output_img, f"Saved: {saved_frames_count} frames", 
                               (width-280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(combined_output_img, f"Aligned: {aligned_timestamp:.3f}", 
                               (width-280, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(combined_output_img, f"IMU base: {start_recording_time:.3f}", 
                               (width-280, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    # 显示相对时间
                    if 'initial_event_time' in globals() and initial_event_time is not None:
                        relative_time = curr_time - initial_event_time
                        cv2.putText(combined_output_img, f"Rel time: {relative_time:.3f}s", 
                                   (width-280, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # 如果没有检测到bbox，仍然需要发布图像消息
            if not current_bbox_info:
                if args.bag_file_path and should_start_saving:
                    # 使用对齐的时间戳
                    if 'initial_event_time' not in globals() or initial_event_time is None:
                        initial_event_time = curr_time
                    
                    relative_time = curr_time - initial_event_time
                    aligned_timestamp = start_recording_time + relative_time
                    aligned_ros_time = rospy.Time.from_sec(aligned_timestamp)
                    
                    cd_output_img_msg = bridge.cv2_to_imgmsg(cd_output_img, encoding="bgr8")
                    cd_output_img_msg.header.stamp = aligned_ros_time
                    cd_output_img_msg.header.frame_id = "image_frame"
                    cd_output_img_pub.publish(cd_output_img_msg)
                else:
                    # 使用当前时间
                    cd_output_img_msg = bridge.cv2_to_imgmsg(cd_output_img, encoding="bgr8")
                    cd_output_img_msg.header.stamp = rospy.Time.now()
                    cd_output_img_msg.header.frame_id = "image_frame"
                    cd_output_img_pub.publish(cd_output_img_msg)

            current_time = time.time()
            skip_frame = int((current_time + 0.005 - start_frame_time)*1000000//args.dt_step)

            # 更新track_list
            update_track_list = []
            if len(track_list) > 0:
                for track_obj in track_list:
                    if(curr_time - track_obj[4] < 0.40):
                        update_track_list.append(track_obj)
            track_list = update_track_list

            window.show(combined_output_img)

            flow_output_img.fill(0)
            frame_count += 1

            if args.out_video:
                writer.writeFrame(combined_output_img.astype(np.uint8)[..., ::-1])

            if window.should_close():
                break

    # 修改：程序结束时显示详细的时间对齐信息
    if bag_writer:
        bag_writer.close()
        output_bag_path = args.bag_file_path.replace('.bag', '_with_events.bag')
        print(f"\n=== Bag Merge Completed ===")
        print(f"Original bag file: {args.bag_file_path}")
        print(f"Output bag file: {output_bag_path}")
        print(f"Total frames processed: {frame_count}")
        print(f"Event data frames saved: {saved_frames_count}")
        print(f"Time alignment info:")
        print(f"  - IMU start time (base): {start_recording_time:.6f}")
        if 'initial_event_time' in globals() and initial_event_time is not None:
            print(f"  - Initial event time: {initial_event_time:.6f}")
            print(f"  - Event data duration: {curr_time - initial_event_time:.6f} seconds")
        print(f"  - Event timestamps aligned relative to IMU start time")
        print(f"Added topics:")
        print(f"  - /event_camera/time_surface (sensor_msgs/Image)")
        print(f"  - /event_camera/bbox_data (std_msgs/String)")
        print(f"Original topics preserved:")
        print(f"  - /livox/imu (sensor_msgs/Imu)")
        print(f"  - /livox/lidar (livox_ros_driver2/CustomMsg)")
        
        # 验证输出文件并显示时间范围
        try:
            with rosbag.Bag(output_bag_path, 'r') as verify_bag:
                topics = verify_bag.get_type_and_topic_info()[1]
                print(f"\nFinal bag contains {len(topics)} topics:")
                
                # 分别统计各个话题的时间范围
                topic_time_ranges = {}
                for topic, msg, t in verify_bag.read_messages():
                    timestamp = t.to_sec()
                    if topic not in topic_time_ranges:
                        topic_time_ranges[topic] = {'start': timestamp, 'end': timestamp, 'count': 0}
                    topic_time_ranges[topic]['end'] = timestamp
                    topic_time_ranges[topic]['count'] += 1
                
                for topic_name, topic_info in topics.items():
                    time_range = topic_time_ranges.get(topic_name, {})
                    duration = time_range.get('end', 0) - time_range.get('start', 0)
                    print(f"  - {topic_name}: {topic_info.message_count} msgs ({topic_info.msg_type})")
                    print(f"    Time range: {time_range.get('start', 0):.3f} - {time_range.get('end', 0):.3f} (duration: {duration:.3f}s)")
                    
        except Exception as e:
            print(f"Error verifying output bag: {e}")

    if args.out_video:
        writer.close()
    dump_estimated_flow_data(args, width, height, all_flow_events,
                             all_dense_flows_start_ts, all_dense_flows_end_ts, all_dense_flows)

if __name__ == "__main__":
    main()