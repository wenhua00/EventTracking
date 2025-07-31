# # Copyright (c) Prophesee S.A. - All Rights Reserved
# #
# # Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# # You may not use this file except in compliance with these License T&C's.
# # A copy of these License T&C's is located in the "licensing" folder accompanying this file.

# """
# Code sample showing how to use Metavision SDK to display results of dense optical flow.
# This version is modified to run without a GUI, suitable for SSH environments.
# """

# import numpy as np
# import os
# import h5py
# import math
# import cv2
# import sys
# import time
# # Required to import metavision_core
# sys.path.append("/usr/lib/python3/dist-packages/")

# from metavision_core.event_io import EventsIterator
# from metavision_core.event_io.raw_reader import initiate_device
# from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
# from metavision_sdk_core import OnDemandFrameGenerationAlgorithm
# from metavision_sdk_cv import PlaneFittingFlowAlgorithm, TimeGradientFlowAlgorithm, TripletMatchingFlowAlgorithm, \
#     DenseFlowFrameGeneratorAlgorithm, SpatioTemporalContrastAlgorithm
# # [MODIFICATION] Removed UI imports as they are not needed for headless operation.
# # from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent
# import rospy
# from std_msgs.msg import String
# from skvideo.io import FFmpegWriter
# from enum import Enum
# from sensor_msgs.msg import Image
# from sensor_msgs.msg import CompressedImage
# from cv_bridge import CvBridge


# bias_diff_off = -0
# bias_diff_on = 0
# bias_hpf = 0
# bias_fo = 0
# bias_refr = 0
# cnt_evs = 0.0
# early_warning_time = 0.0400
# last_frame_time = 0

# track_list = []  # x y w h nowtime evsnumber start_time track_id
# cnt_track_id = 0


# class FlowType(Enum):
#     PlaneFitting = "PlaneFitting"
#     TimeGradient = "TimeGradient"
#     TripletMatching = "TripletMatching"

#     def __str__(self):
#         return self.value


# def parse_args():
#     import argparse
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description='Metavision Dense Optical Flow sample.',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     input_group = parser.add_argument_group(
#         "Input", "Arguments related to input sequence.")
#     input_group.add_argument(
#         '-i', '--input-event-file', dest='event_file_path', default="",
#         help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
#         "If it's a camera serial number, it will try to open that camera instead.33333")
#     input_group.add_argument(
#         '--replay_factor', type=float, default=1,
#         help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")
#     input_group.add_argument(
#         '--dt-step', type=int, default=20000, dest="dt_step",
#         help="Time processing step (in us), used as iteration delta_t period, visualization framerate and accumulation time.")

#     algo_settings_group = parser.add_argument_group(
#         "Algo Settings", "Arguments related to algorithmic configuration.")
#     algo_settings_group.add_argument(
#         "--flow-type", dest="flow_type", type=FlowType, choices=list(FlowType),
#         default=FlowType.TripletMatching, help="Chosen type of dense flow algorithm to run")
#     algo_settings_group.add_argument(
#         "-r", "--receptive-field-radius", dest="receptive_field_radius", type=float, default=3,  # default 3
#         help="Radius of the receptive field used for flow estimation, in pixels.")
#     algo_settings_group.add_argument("--min-flow", dest="min_flow_mag", type=float,  default=3500.,  # default 1000
#                                      help="Minimum observable flow magnitude, in px/s.")
#     algo_settings_group.add_argument("--max-flow", dest="max_flow_mag", type=float,  default=6500.,
#                                      help="Maximum observable flow magnitude, in px/s.")
#     algo_settings_group.add_argument("--stc-filter-thr", dest="stc_filter_thr", type=int,  default=10000,
#                                      help="Length of the time window for filtering (in us).")
#     algo_settings_group.add_argument(
#         "--visu-scale", dest="visualization_flow_scale", type=float, default=0.8,
#         help="Flow magnitude used to scale the upper bound of the flow visualization, in px/s. If negative, will use 1/5-th of maximum flow magnitude.")

#     output_flow_group = parser.add_argument_group(
#         "Output flow", "Arguments related to output optical flow.")
#     output_flow_group.add_argument(
#         "--output-sparse-npy-filename", dest="output_sparse_npy_filename",
#         help="If provided, the predictions will be saved as numpy structured array of EventOpticalFlow. In this "
#         "format, the flow vx and vy are expressed in pixels per second.")
#     output_flow_group.add_argument(
#         "--output-dense-h5-filename", dest="output_dense_h5_filename",
#         help="If provided, the predictions will be saved as a sequence of dense flow in HDF5 data. The flows are "
#         "averaged pixelwise over timeslices of --dt-step. The dense flow is expressed in terms of "
#         "pixels per timeslice (of duration dt-step), not in pixels per second.")
#     output_flow_group.add_argument(
#         '-o', '--out-video', dest='out_video', type=str, default="",
#         help="Path to an output AVI file to save the resulting video.")
#     output_flow_group.add_argument(
#         '--fps', dest='fps', type=int, default=30,
#         help="replay fps of output video")
#     output_flow_group.add_argument(
#         '--no-legend', dest='show_legend', action='store_false',
#         help="Set to remove the legend from the display")

#     args = parser.parse_args()

#     if args.output_sparse_npy_filename:
#         assert not os.path.exists(args.output_sparse_npy_filename)
#     if args.output_dense_h5_filename:
#         assert not os.path.exists(args.output_dense_h5_filename)
#     if args.visualization_flow_scale <= 0:
#         args.visualization_flow_scale = 1

#     return args


# def accumulate_estimated_flow_data(
#         args, width, height, processing_ts, flow_buffer, all_flow_events, all_dense_flows_start_ts,
#         all_dense_flows_end_ts, all_dense_flows):
#     """Accumulates estimated flow data in buffers to dump estimation results"""
#     if args.output_sparse_npy_filename:
#         all_flow_events.append(flow_buffer.numpy().copy())
#     if args.output_dense_h5_filename:
#         all_dense_flows_start_ts.append(
#             processing_ts - args.dt_step)
#         all_dense_flows_end_ts.append(processing_ts)
#         flow_np = flow_buffer.numpy()
#         if flow_np.size == 0:
#             all_dense_flows.append(
#                 np.zeros((2, height, width), dtype=np.float32))
#         else:
#             xs, ys, vx, vy = flow_np["x"], flow_np["y"], flow_np["vx"], flow_np["vy"]
#             coords = np.stack((ys, xs))
#             abs_coords = np.ravel_multi_index(coords, (height, width))
#             counts = np.bincount(abs_coords, weights=np.ones(flow_np.size),
#                                  minlength=height*width).reshape(height, width)
#             flow_x = np.bincount(
#                 abs_coords, weights=vx, minlength=height*width).reshape(height, width)
#             flow_y = np.bincount(
#                 abs_coords, weights=vy, minlength=height*width).reshape(height, width)
#             mask_multiple_events = counts > 1
#             flow_x[mask_multiple_events] /= counts[mask_multiple_events]
#             flow_y[mask_multiple_events] /= counts[mask_multiple_events]

#             # flow expressed in pixels per delta_t
#             flow_x *= args.dt_step * 1e-6
#             flow_y *= args.dt_step * 1e-6
#             flow = np.stack((flow_x, flow_y)).astype(np.float32)
#             all_dense_flows.append(flow)


# def dump_estimated_flow_data(
#         args, width, height, all_flow_events, all_dense_flows_start_ts, all_dense_flows_end_ts, all_dense_flows):
#     """Write accumulated flow results to output files"""
#     try:
#         if args.output_sparse_npy_filename:
#             print("Writing output file: ", args.output_sparse_npy_filename)
#             all_flow_events = np.concatenate(all_flow_events)
#             np.save(args.output_sparse_npy_filename, all_flow_events)
#         if args.output_dense_h5_filename:
#             print("Writing output file: ", args.output_dense_h5_filename)
#             flow_start_ts = np.array(all_dense_flows_start_ts)
#             flow_end_ts = np.array(all_dense_flows_end_ts)
#             flows = np.stack(all_dense_flows)
#             N = flow_start_ts.size
#             assert flow_end_ts.size == N
#             assert flows.shape == (N, 2, height, width)
#             dirname = os.path.dirname(args.output_dense_h5_filename)
#             if not os.path.isdir(dirname):
#                 os.makedirs(dirname)
#             flow_h5 = h5py.File(args.output_dense_h5_filename, "w")
#             flow_h5.create_dataset(
#                 "flow_start_ts", data=flow_start_ts, compression="gzip")
#             flow_h5.create_dataset(
#                 "flow_end_ts", data=flow_end_ts, compression="gzip")
#             flow_h5.create_dataset("flow", data=flows.astype(
#                 np.float32), compression="gzip")
#             flow_h5["flow"].attrs["input_file_name"] = os.path.basename(
#                 args.event_file_path)
#             flow_h5["flow"].attrs["checkpoint_path"] = "metavision_dense_optical_flow"
#             flow_h5["flow"].attrs["event_input_height"] = height
#             flow_h5["flow"].attrs["event_input_width"] = width
#             flow_h5["flow"].attrs["delta_t"] = args.dt_step
#             flow_h5.close()
#     except Exception as e:
#         print(e)
#         raise


# def calculate_diou(bbox1, bbox2):
#     # 提取第一个边界框的坐标
#     x1_min, y1_min = bbox1[0], bbox1[1]
#     x1_max, y1_max = x1_min + bbox1[2], y1_min + bbox1[3]
#     # 提取第二个边界框的坐标
#     x2_min, y2_min = bbox2[0], bbox2[1]
#     x2_max, y2_max = x2_min + bbox2[2], y2_min + bbox2[3]
#     # 计算交集区域的坐标
#     inter_x_min = max(x1_min, x2_min)
#     inter_y_min = max(y1_min, y2_min)
#     inter_x_max = min(x1_max, x2_max)
#     inter_y_max = min(y1_max, y2_max)
#     # 交集的宽度和高度
#     inter_width = max(0, inter_x_max - inter_x_min)
#     inter_height = max(0, inter_y_max - inter_y_min)
#     # 交集区域面积
#     inter_area = inter_width * inter_height
#     # 第一个边界框的面积
#     bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
#     # 第二个边界框的面积
#     bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
#     # 并集面积 = 两个面积之和 - 交集面积
#     union_area = bbox1_area + bbox2_area - inter_area
#     # IoU 计算
#     iou = inter_area / union_area if union_area > 0 else 0
#     # 计算边界框的中心点
#     bbox1_center = ((x1_min + x1_max) / 2, (y1_min + y1_max) / 2)
#     bbox2_center = ((x2_min + x2_max) / 2, (y2_min + y2_max) / 2)
#     # 计算中心点之间的欧几里得距离
#     rho = np.sqrt((bbox1_center[0] - bbox2_center[0])
#                   ** 2 + (bbox1_center[1] - bbox2_center[1]) ** 2)
#     # 计算包围两个边界框的最小矩形的对角线长度
#     enclosing_x_min = min(x1_min, x2_min)
#     enclosing_y_min = min(y1_min, y2_min)
#     enclosing_x_max = max(x1_max, x2_max)
#     enclosing_y_max = max(y1_max, y2_max)
#     c = np.sqrt((enclosing_x_max - enclosing_x_min) **
#                 2 + (enclosing_y_max - enclosing_y_min) ** 2)
#     # DIoU 计算
#     diou = iou - (rho ** 2) / (c ** 2)
#     return diou


# def track_(bbox):
#     global cnt_track_id, track_list
#     track_id = None
#     max_diou = -1
#     if (len(track_list) == 0):
#         cnt_track_id = cnt_track_id + 1
#         track_list.append(bbox + [bbox[4], cnt_track_id])
#         return track_list[-1], False
#     else:
#         for i in range(len(track_list)):
#             diou = calculate_diou(track_list[i][:-1], bbox)
#             if (diou > -0.2 and diou > max_diou):
#                 max_diou = diou
#                 track_id = i
#         if (track_id is not None):
#             track_list[track_id][:6] = bbox
#             flag = False
#             if (track_list[track_id][4] - track_list[track_id][-2] >= early_warning_time):
#                 flag = True
#             return track_list[track_id], flag
#         else:
#             cnt_track_id = cnt_track_id + 1
#             track_list.append(bbox + [bbox[4], cnt_track_id])
#             return track_list[-1], False

# # [MODIFICATION] The main function is refactored to remove all GUI components.
# def main():
#     """ Main """
#     args = parse_args()

#     # ROS node initialization
#     # The `disable_signals=True` argument can sometimes help, but checking is_shutdown() is the canonical way.
#     # We will stick to the canonical way for robustness.
#     rospy.init_node('bbox_publisher', anonymous=True)
#     bbox_pub = rospy.Publisher('/bbox/'+args.event_file_path[-5:], String, queue_size=10)
#     filtered_points_pub = rospy.Publisher('/filtered_points/'+args.event_file_path[-5:], String, queue_size=10)
#     bridge = CvBridge()
#     cd_output_img_pub = rospy.Publisher('/cd_output_img_topic/'+args.event_file_path[-5:], Image, queue_size=3)
    
#     device = initiate_device(path=args.event_file_path)

#     # 此处设置相机同步模式为主设备
#     device.get_i_camera_synchronization().set_mode_master()

#     if device.get_i_ll_biases():
#         device.get_i_ll_biases().set("bias_diff_off", bias_diff_off)
#         device.get_i_ll_biases().set("bias_diff_on", bias_diff_on)
#         device.get_i_ll_biases().set("bias_hpf", bias_hpf)
#         device.get_i_ll_biases().set("bias_fo", bias_fo)
#         device.get_i_ll_biases().set("bias_refr", bias_refr)

#     mv_iterator = EventsIterator.from_device(device=device, delta_t=args.dt_step)

#     # Set ERC to 20Mev/s
#     if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device and mv_iterator.reader.device.get_i_erc_module():
#         erc_module = mv_iterator.reader.device.get_i_erc_module()
#         erc_module.set_cd_event_rate(20000000)
#         erc_module.enable(True)

#     if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
#         mv_iterator = LiveReplayEventsIterator(
#             mv_iterator, replay_factor=args.replay_factor)
#     height, width = mv_iterator.get_size()

#     # ... (rest of the setup code remains the same) ...
#     event_frame_gen = OnDemandFrameGenerationAlgorithm(width, height, args.dt_step)
#     if args.flow_type == FlowType.PlaneFitting:
#         radius = math.floor(args.receptive_field_radius)
#         print(f"Instantiating PlaneFittingFlowAlgorithm with radius={radius}")
#         flow_processor = PlaneFittingFlowAlgorithm(width, height, radius, -1)
#     elif args.flow_type == FlowType.TimeGradient:
#         radius = int(args.receptive_field_radius)
#         print(f"Instantiating TimeGradientFlowAlgorithm with radius={radius}, min_flow={args.min_flow_mag}, bit_cut=2")
#         flow_processor = TimeGradientFlowAlgorithm(width, height, radius, args.min_flow_mag, 2)
#     else:
#         radius = 0.5 * args.receptive_field_radius
#         print(f"Instantiating TripletMatchingFlowAlgorithm with radius={radius}, min_flow={args.min_flow_mag}, max_flow={args.max_flow_mag}")
#         flow_processor = TripletMatchingFlowAlgorithm(width, height, radius, args.min_flow_mag, args.max_flow_mag)
#     flow_buffer = TripletMatchingFlowAlgorithm.get_empty_output_buffer()
#     flow_frame_gen = DenseFlowFrameGeneratorAlgorithm(
#         width, height, args.max_flow_mag, args.visualization_flow_scale,
#         DenseFlowFrameGeneratorAlgorithm.VisualizationMethod.Arrows,
#         DenseFlowFrameGeneratorAlgorithm.AccumulationPolicy.Average)
#     print(f"Instantiating SpatioTemporalContrastAlgorithm with thresh={args.stc_filter_thr}")
#     stc_filter = SpatioTemporalContrastAlgorithm(width, height, args.stc_filter_thr, True)
#     events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()
#     all_flow_events, all_dense_flows, all_dense_flows_start_ts, all_dense_flows_end_ts = [], [], [], []
#     writer = None
#     if args.out_video:
#         video_name = args.out_video + ".avi"
#         print(f"Will write output video to: {video_name}")
#         writer = FFmpegWriter(video_name, inputdict={'-r': str(args.fps)}, outputdict={
#             '-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-r': str(args.fps)})
#     cd_output_img = np.zeros((height, width, 3), np.uint8)
#     flow_output_img = np.zeros((height, width, 3), np.uint8)
#     combined_output_img = np.zeros((height, width, 3), np.uint8)
#     processing_ts = mv_iterator.start_ts
#     global track_list
#     last_time, FPS, cnt_fps, cnt_time = 0.0, 0, 0, 0.0
#     global last_frame_time
#     skip_frame, bbox_msg, flag_bbox_msg, curr_time = 0, None, False, None
    
#     try:
#         # Process events
#         for evs in mv_iterator:
            
#             # [MODIFICATION] ADD THIS CHECK AT THE BEGINNING OF THE LOOP
#             if rospy.is_shutdown():
#                 print("ROS shutdown request received. Exiting loop.")
#                 break

#             processing_ts += mv_iterator.delta_t
#             start_frame_time = time.time()
            
#             # Filter Events using STC
#             stc_filter_start = time.time()
#             stc_filter.process_events(evs, events_buf)
#             stc_filter_end = time.time()

#             if not events_buf:
#                 continue

#             if (skip_frame):
#                 skip_frame = skip_frame - 1
#                 continue

#             # ... (the rest of the loop logic remains exactly the same) ...
#             event_frame_gen.process_events(events_buf)
#             event_frame_gen.generate(processing_ts, cd_output_img)
#             flow_start = time.time()
#             flow_processor.process_events(events_buf, flow_buffer)
#             accumulate_estimated_flow_data(
#                 args, width, height, processing_ts, flow_buffer, all_flow_events, all_dense_flows_start_ts,
#                 all_dense_flows_end_ts, all_dense_flows)
#             flow_end = time.time()
#             flow_frame_gen.process_events(flow_buffer)
#             flow_frame_gen.generate(flow_output_img)
#             curr_time = start_frame_time
#             cnt_time += curr_time - last_time
#             cnt_fps += 1
#             if(cnt_time >= 1):
#                 FPS = cnt_fps
#                 cnt_time = 0
#                 cnt_fps = 0
#                 print(f"Processing at approximately {FPS} FPS")
#             last_time = curr_time
#             cv2.addWeighted(cd_output_img, 0.6, flow_output_img, 0.4, 0, combined_output_img)
#             flag_bbox_msg = False
#             bbox_msg = None
#             tracking_start = time.time()
#             if(len(flow_buffer.numpy()) > 0):
#                 xyt_selected = np.column_stack((flow_buffer.numpy()['x'][:], flow_buffer.numpy()['y'][:], flow_buffer.numpy()['t'][:]))
#                 xy_selected = np.column_stack((flow_buffer.numpy()['x'][:], flow_buffer.numpy()['y'][:]))
#                 y_coords = xy_selected[:, 1].astype(int)
#                 x_coords = xy_selected[:, 0].astype(int)
#                 non_black_mask = ~np.all(flow_output_img[y_coords, x_coords] == [0, 0, 0], axis=1)
#                 filtered_points = xy_selected[non_black_mask]
#                 filtered_points_with_time = xyt_selected[non_black_mask]
#                 combined_output_img[filtered_points[:, 1].astype(int), filtered_points[:, 0].astype(int)] = [0, 255, 255]
#                 filtered_points = np.array(filtered_points)
#                 avg_vx, avg_vy = 0, 0
#                 vx = flow_buffer.numpy()['vx'][:]
#                 vy = flow_buffer.numpy()['vy'][:]
#                 avg_vx = np.mean(vx)
#                 avg_vy = np.mean(vy)
#                 if filtered_points.size > 5:
#                     Q1 = np.percentile(filtered_points, 25, axis=0)
#                     Q3 = np.percentile(filtered_points, 75, axis=0)
#                     IQR = Q3 - Q1
#                     lower_bound = Q1 - 1.5 * IQR
#                     upper_bound = Q3 + 1.5 * IQR
#                     filtered_points = filtered_points[
#                         (filtered_points[:, 0] >= lower_bound[0]) & (filtered_points[:, 0] <= upper_bound[0]) &
#                         (filtered_points[:, 1] >= lower_bound[1]) & (filtered_points[:, 1] <= upper_bound[1])
#                     ]
#                     if filtered_points.size > 0:
#                         min_x, min_y = filtered_points.min(axis=0)
#                         max_x, max_y = filtered_points.max(axis=0)
#                         bbox_x, bbox_y = (max_x + min_x)//2, (max_y + min_y)//2
#                         bbox_width, bbox_heigth = max_x - min_x, max_y - min_y
#                         wh_ratio = bbox_width/bbox_heigth if bbox_width > bbox_heigth else bbox_heigth/bbox_width
#                         if(wh_ratio < 5.5 and min(bbox_width, bbox_heigth) < 500 and min(bbox_width, bbox_heigth) > 2):
#                             bbox = [bbox_x, bbox_y, bbox_width, bbox_heigth, curr_time, filtered_points.size]
#                             track_bbox, flag = track_(bbox)
#                             if(flag):
#                                 cv2.rectangle(combined_output_img, (min_x, min_y), (max_x, max_y), color=(0, 0, 255), thickness=6)
#                             else:
#                                 cv2.rectangle(combined_output_img, (min_x, min_y), (max_x, max_y), color=(0, 255, 0), thickness=5)
#                             bbox_msg = f"{bbox_x},{bbox_y},{bbox_width},{bbox_heigth},{filtered_points.size},{avg_vx},{avg_vy}"
#                             bbox_pub.publish(bbox_msg)
#                             points_msg = '\n'.join([f"{int(x)},{int(y)},{int(t)}" for x, y, t in filtered_points_with_time])
#                             filtered_points_pub.publish(points_msg)
#                             cv2.putText(combined_output_img, str(track_bbox[-1]), (min_x, min_y-10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
#                             flag_bbox_msg = True
#             tracking_end = time.time()
#             cv2.putText(combined_output_img, "FPS: "+str(FPS), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)
#             current_time = time.time()
#             skip_frame = int((current_time + 0.005 - start_frame_time)*1000000//args.dt_step)
#             update_track_list = [track_obj for track_obj in track_list if curr_time - track_obj[4] < 0.40]
#             track_list = update_track_list
#             cd_output_img_msg = bridge.cv2_to_imgmsg(combined_output_img, encoding="bgr8")
#             cd_output_img_msg.header.stamp = rospy.Time.now()
#             cd_output_img_msg.header.frame_id = "image_frame"
#             cd_output_img_pub.publish(cd_output_img_msg)
#             if flag_bbox_msg:
#                 print(f"STC: {(stc_filter_end - stc_filter_start)*1000:.2f}ms | Flow: {(flow_end - flow_start)*1000:.2f}ms | Track: {(tracking_end - tracking_start)*1000:.2f}ms")
#             flow_output_img.fill(0)
#             if writer:
#                 writer.writeFrame(combined_output_img.astype(np.uint8)[..., ::-1])

#     except KeyboardInterrupt:
#         # This block might still be useful if an interrupt occurs before the loop starts
#         # or in a part of the code not governed by ROS signals. It's good practice to keep it.
#         print("\nProcess interrupted by user (KeyboardInterrupt). Cleaning up...")
#     finally:
#         if writer:
#             print("Closing video writer...")
#             writer.close()
#         dump_estimated_flow_data(args, width, height, all_flow_events,
#                                  all_dense_flows_start_ts, all_dense_flows_end_ts, all_dense_flows)
#         print("Processing finished.")

# if __name__ == "__main__":
#     main()
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

    # ROS node initialization
    rospy.init_node('bbox_publisher', anonymous=True)
    bbox_pub = rospy.Publisher('/bbox/'+args.event_file_path[-5:], String, queue_size=10)
    filtered_points_pub = rospy.Publisher('/filtered_points/'+args.event_file_path[-5:], String, queue_size=10)
    bridge = CvBridge()
    cd_output_img_pub = rospy.Publisher('/cd_output_img_topic/'+args.event_file_path[-5:], Image, queue_size=3)
    # compressed_img_pub = rospy.Publisher('/cd_output_img_topic/'+args.event_file_path[-5:]+'/compressed', CompressedImage, queue_size=1)

    device = initiate_device(path=args.event_file_path)

    #此处设置相机同步模式为主设备
    # device.get_i_camera_synchronization().set_mode_master()

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
        # erc_module.enable(True)
        erc_module.enable(False)


    if args.replay_factor > 0 and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(
            mv_iterator, replay_factor=args.replay_factor)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Event Frame Generator
    event_frame_gen = OnDemandFrameGenerationAlgorithm(
        width, height, args.dt_step)

    # Dense Optical Flow Algorithm
    # The input receptive field radius represents the total area of the neighborhood that is used to estimate flow. We
    # use an algorithm-dependent heuristic to convert this into the search radius to be used for each algorithm.
    if args.flow_type == FlowType.PlaneFitting:
        radius = math.floor(args.receptive_field_radius)
        print(
            f"Instantiating PlaneFittingFlowAlgorithm with radius={radius}")
        flow_algo_planefitting = PlaneFittingFlowAlgorithm(
            width, height, radius, -1)
        flow_algo_timegradient = None
        flow_algo_tripletmatching = None
    elif args.flow_type == FlowType.TimeGradient:
        radius = int(args.receptive_field_radius)
        print(
            f"Instantiating TimeGradientFlowAlgorithm with radius={radius}, min_flow={args.min_flow_mag}, bit_cut=2")
        flow_algo_timegradient = TimeGradientFlowAlgorithm(width, height, radius, args.min_flow_mag, 2)
        flow_algo_planefitting = None
        flow_algo_tripletmatching = None
    else:
        radius = 0.5*args.receptive_field_radius
        print(
            f"Instantiating TripletMatchingFlowAlgorithm with radius={radius}, min_flow={args.min_flow_mag}, max_flow={args.max_flow_mag}")
        flow_algo_tripletmatching = TripletMatchingFlowAlgorithm(
            width, height, radius, args.min_flow_mag, args.max_flow_mag)
        flow_algo_planefitting = None
        flow_algo_timegradient = None
    flow_buffer = TripletMatchingFlowAlgorithm.get_empty_output_buffer()

    # Dense Flow Frame Generator
    flow_frame_gen = DenseFlowFrameGeneratorAlgorithm(
        width, height, args.max_flow_mag, args.visualization_flow_scale,
        DenseFlowFrameGeneratorAlgorithm.VisualizationMethod.Arrows,
        DenseFlowFrameGeneratorAlgorithm.AccumulationPolicy.Average)
    flow_legend_img = np.zeros((100, 100, 3), np.uint8)
    flow_frame_gen.generate_legend_image(flow_legend_img)
    legend_mask = flow_legend_img != 0

    # STC filter
    print(
        f"Instantiating SpatioTemporalContrastAlgorithm with thresh={args.stc_filter_thr}")
    stc_filter = SpatioTemporalContrastAlgorithm(
        width, height, args.stc_filter_thr, True)
    events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()

    all_flow_events = []
    all_dense_flows = []
    all_dense_flows_start_ts = []
    all_dense_flows_end_ts = []

    # Window - Graphical User Interface

    if args.out_video:
        video_name = args.out_video + ".avi"
        writer = FFmpegWriter(video_name, inputdict={'-r': str(args.fps)}, outputdict={
            '-vcodec': 'libx264',
            '-pix_fmt': 'yuv420p',
            '-r': str(args.fps)
        })

    # def keyboard_cb(key, scancode, action, mods):
    #     if action != UIAction.RELEASE:
    #         return
    #     if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
    #         window.set_close_flag()

    # window.set_keyboard_callback(keyboard_cb)

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
    global track_list
    last_time = 0.0
    FPS = 0
    cnt_fps = 0
    cnt_time = 0.0
    global last_frame_time
    skip_frame = 0
    for evs in mv_iterator:
        # evs = filter_foreground_events(evs, min_count=2)
        processing_ts += mv_iterator.delta_t

        start_frame_time = time.time()
        # Dispatch system events to the window
        EventLoop.poll_and_dispatch()

        # Filter Events using STC
        # stc_filter.process_events(evs, events_buf)
        events_buf = evs
        
        evs_number = len(evs)
        if(evs_number == 0):
            continue

        #print("org:",len(evs))

        if(skip_frame):
            skip_frame = skip_frame - 1
            continue

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

        # print(flow_buffer.numpy()[['x', 'y', 'vx', 'vy', 't']])
        
        curr_time = start_frame_time
        cnt_time = cnt_time + curr_time - last_time
        cnt_fps = cnt_fps + 1
        if(cnt_time>=1):
            FPS = cnt_fps
            cnt_time = 0
            cnt_fps = 0
        last_time = curr_time
        
        # Update the display
        cv2.addWeighted(cd_output_img, 0.6, flow_output_img, 0.4, 0, combined_output_img)

        if(len(flow_buffer.numpy())>0):
            
            xyt_selected = np.column_stack((flow_buffer.numpy()['x'][:], flow_buffer.numpy()['y'][:], flow_buffer.numpy()['t'][:]))
            xy_selected = np.column_stack((flow_buffer.numpy()['x'][:], flow_buffer.numpy()['y'][:]))
        # 提取 y 和 x 坐标
            y_coords = xy_selected[:, 1].astype(int)
            x_coords = xy_selected[:, 0].astype(int)

            # 检查 flow_output_img 中 (y, x) 位置是否为 [0, 0, 0]
            # 先提取所有对应位置的像素值，然后判断是否为 [0, 0, 0]
            non_black_mask = ~np.all(flow_output_img[y_coords, x_coords] == [0, 0, 0], axis=1)

            # 筛选出满足条件的点
            filtered_points = xy_selected[non_black_mask]
            filtered_points_with_time = xyt_selected[non_black_mask]

            # 在 combined_output_img 上绘制圆点（矢量化操作）
            combined_output_img[filtered_points[:, 1].astype(int), filtered_points[:, 0].astype(int)] = [0, 255, 255]

            # 转换为 NumPy 数组
            # filtered_points = xy_selected
            filtered_points = np.array(filtered_points)

            # 检查是否有满足条件的点
            # print(filtered_points.size)
            if filtered_points.size > 5:
                # 计算四分位数
                Q1 = np.percentile(filtered_points, 25, axis=0)
                Q3 = np.percentile(filtered_points, 75, axis=0)
                IQR = Q3 - Q1

                # 定义上下限
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # 过滤掉离群点
                filtered_points = filtered_points[
                    (filtered_points[:, 0] >= lower_bound[0]) & (filtered_points[:, 0] <= upper_bound[0]) &
                    (filtered_points[:, 1] >= lower_bound[1]) & (filtered_points[:, 1] <= upper_bound[1])
                ]

                # 重新计算最小值和最大值
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

                            # 发布bbox和平均vx,vy
                            bbox_msg = f"{bbox_x},{bbox_y},{bbox_width},{bbox_heigth},{filtered_points_number},{mean_vx:.2f},{mean_vy:.2f}"
                            bbox_pub.publish(bbox_msg)
                            points_msg = '\n'.join([f"{int(x)},{int(y)},{int(t)}" for x, y, t in filtered_points_with_time])
                            filtered_points_pub.publish(points_msg)
                            cv2.putText(combined_output_img, str(track_bbox[-1]), (min_x, min_y-10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
                    else:
                        # 处理无效边界框（跳过或标记）
                        continue  # 或 wh_ratio = float('inf') 根据业务逻辑调整


        cv2.putText(combined_output_img, "FPS: "+str(FPS), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 5)

        current_time = time.time()
        skip_frame = int((current_time + 0.005 - start_frame_time)*1000000//args.dt_step)

        update_track_list = []
        if len(track_list) > 0:
            for track_obj in track_list:
                if(curr_time - track_obj[4] < 0.40):
                    update_track_list.append(track_obj)
        track_list = update_track_list

        # cd_output_img_rgb = cv2.cvtColor(cd_output_img, cv2.COLOR_BGR2RGB)
        cd_output_img_msg = bridge.cv2_to_imgmsg(cd_output_img, encoding="bgr8")
        # 设置图像消息的时间戳，使用当前时间
        cd_output_img_msg.header.stamp = rospy.Time.now()
        # 设置图像消息的帧 ID，可以根据需要修改
        cd_output_img_msg.header.frame_id = "image_frame"
        cd_output_img_pub.publish(cd_output_img_msg)
        # compressed_img_msg = bridge.cv2_to_compressed_imgmsg(cd_output_img)
        # compressed_img_pub.publish(compressed_img_msg)

        # window.show(combined_output_img)
        # window.show(flow_output_img)


        flow_output_img.fill(0)

        if args.out_video:
            writer.writeFrame(combined_output_img.astype(np.uint8)[..., ::-1])


    if args.out_video:
        writer.close()
    dump_estimated_flow_data(args, width, height, all_flow_events,
                             all_dense_flows_start_ts, all_dense_flows_end_ts, all_dense_flows)


if __name__ == "__main__":
    main() 
