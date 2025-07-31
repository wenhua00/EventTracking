# from metavision_core.event_io.raw_reader import initiate_device
# from metavision_core.event_io import EventsIterator
# from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
# from metavision_sdk_cv import SpatioTemporalContrastAlgorithm
# from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent
# import argparse
# import time
# import os

# def parse_args():
#     parser = argparse.ArgumentParser(description='Metavision RAW file Recorder with ERC+STC filters.',
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument(
#         '-o', '--output-dir', default="", help="Directory where to create RAW file with recorded event data")
#     parser.add_argument("--stc-filter-thr", dest="stc_filter_thr", type=int, default=30000,
#                         help="时间窗阈值（微秒）用于 STC 滤波")
#     return parser.parse_args()

# def main():
#     args = parse_args()

#     # 初始化设备
#     device = initiate_device("")

#     # 启动录制
#     if device.get_i_events_stream():
#         log_path = "recording_" + time.strftime("%y%m%d_%H%M%S", time.localtime()) + ".raw"
#         if args.output_dir:
#             log_path = os.path.join(args.output_dir, log_path)
#         print(f"[INFO] Recording to {log_path}")
#         device.get_i_events_stream().log_raw_data(log_path)

#     # 事件迭代器
#     mv_iterator = EventsIterator.from_device(device=device)
#     height, width = mv_iterator.get_size()

#     # ERC 设置
#     if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device and mv_iterator.reader.device.get_i_erc_module():
#         erc_mod = mv_iterator.reader.device.get_i_erc_module()
#         erc_mod.set_cd_event_rate(20000000)  # 20 Mev/s
#         erc_mod.enable(True)
#         print("[INFO] ERC enabled with 20 Mev/s")

#     # STC 滤波器
#     stc_filter = SpatioTemporalContrastAlgorithm(width, height, args.stc_filter_thr, True)
#     events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()
#     print(f"[INFO] STC filter instantiated with threshold={args.stc_filter_thr} μs")

#     # Frame generator
#     event_frame_gen = PeriodicFrameGenerationAlgorithm(
#         sensor_width=width, sensor_height=height,
#         fps=25, palette=ColorPalette.Dark)

#     # GUI 窗口
#     with MTWindow(title="Filtered Recorder", width=width, height=height,
#                   mode=BaseWindow.RenderMode.BGR) as window:
#         def keyboard_cb(key, scancode, action, mods):
#             if key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
#                 window.set_close_flag()
#         window.set_keyboard_callback(keyboard_cb)

#         def on_frame_cb(ts, cd_frame):
#             window.show_async(cd_frame)
#         event_frame_gen.set_output_callback(on_frame_cb)

#         # 主循环：ERC + STC + 显示 + 录制
#         for evs in mv_iterator:
#             EventLoop.poll_and_dispatch()

#             # STC 滤波
#             stc_filter.process_events(evs, events_buf)
#             filtered_evs = events_buf

#             # 渲染并显示
#             event_frame_gen.process_events(filtered_evs)

#             if window.should_close():
#                 print("[INFO] Stopping recording.")
#                 device.get_i_events_stream().stop_log_raw_data()
#                 break

# if __name__ == "__main__":
#     main()
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_cv import SpatioTemporalContrastAlgorithm
import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Metavision RAW file Recorder with ERC+STC filters.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-o', '--output-dir', default="", help="Directory where to create RAW file with recorded event data")
    parser.add_argument("--stc-filter-thr", dest="stc_filter_thr", type=int, default=30000,
                        help="时间窗阈值（微秒）用于 STC 滤波")
    return parser.parse_args()

def main():
    args = parse_args()

    # 初始化设备
    device = initiate_device("")

    # 启动录制
    if device.get_i_events_stream():
        # 获取 ROS 风格时间戳（单位秒，float）
        ros_timestamp = time.time()
        ros_timestamp_str = f"{ros_timestamp:.6f}"  # 保留微秒

        # 构造文件名（在文件名中加时间戳）
        filename_prefix = f"recording_{ros_timestamp_str}"  # recording_1721720702.123456
        log_path = filename_prefix + ".raw"

        # 如果指定了输出目录，则拼接路径
        if args.output_dir:
            log_path = os.path.join(args.output_dir, log_path)

        # 打印信息
        print(f"[INFO] Recording to {log_path}")
        device.get_i_events_stream().log_raw_data(log_path)

        # 保存 ROS 时间戳到 .txt 文件
        timestamp_txt_path = os.path.splitext(log_path)[0] + "_timestamp.txt"  # 跟 RAW 文件同名
        with open(timestamp_txt_path, "w") as f:
            f.write(ros_timestamp_str + "\n")
        print(f"[INFO] Timestamp saved to {timestamp_txt_path}")

        if args.output_dir:
            log_path = os.path.join(args.output_dir, log_path)
        print(f"[INFO] Recording to {log_path}")
        device.get_i_events_stream().log_raw_data(log_path)

    # 事件迭代器
    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()

    # ERC 设置
    if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device and mv_iterator.reader.device.get_i_erc_module():
        erc_mod = mv_iterator.reader.device.get_i_erc_module()
        erc_mod.set_cd_event_rate(20000000)  # 20 Mev/s
        erc_mod.enable(True)
        print("[INFO] ERC enabled with 20 Mev/s")

    # STC 滤波器设置（与你之前用的一致）
    stc_filter = SpatioTemporalContrastAlgorithm(width, height, args.stc_filter_thr, True)

    events_buf = SpatioTemporalContrastAlgorithm.get_empty_output_buffer()
    print(f"[INFO] STC filter instantiated with length=4, threshold={args.stc_filter_thr} μs")

    # 主循环：ERC + STC
    for evs in mv_iterator:
        # STC 滤波
        stc_filter.process_events(evs, events_buf)
        filtered_evs = events_buf

        # 你可以在此处添加处理 filtered_evs 的逻辑，例如统计信息、保存等
        # 当前版本不显示也不处理，仅录制原始 RAW 文件

if __name__ == "__main__":
    main()
