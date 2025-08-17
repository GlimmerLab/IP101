#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实时视频去雾算法实现

该模块实现了专门针对视频流的实时去雾算法，通过时间一致性优化、
帧间信息利用和高效的处理管道，实现高质量的实时视频去雾效果。

算法特点：
1. 时间一致性：利用帧间相关性保证去雾结果的时间稳定性
2. 实时性能：优化的算法管道支持实时视频处理
3. 自适应参数：根据场景内容自动调整去雾参数
4. 多线程处理：支持并行处理提升性能

作者: GlimmerLab
版本: 1.0.0
"""

import cv2
import numpy as np
import logging
import threading
import queue
import time
from typing import Tuple, Optional, List, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import deque


@dataclass
class RealtimeDehazingParams:
    """实时去雾算法参数配置"""

    # 基础去雾参数
    dark_channel_size: int = 7  # 较小的核尺寸以提升速度
    min_transmission: float = 0.15
    omega: float = 0.9

    # 时间一致性参数
    temporal_weight: float = 0.3  # 时间权重
    history_frames: int = 5  # 历史帧数量

    # 性能优化参数
    resize_factor: float = 0.5  # 处理时的缩放因子
    skip_frames: int = 1  # 跳帧处理间隔

    # 大气光估计参数
    airlight_update_interval: int = 30  # 大气光更新间隔（帧）
    airlight_adaptation_rate: float = 0.1  # 大气光自适应速率

    # 滤波参数
    bilateral_d: int = 5
    bilateral_sigma_color: float = 25
    bilateral_sigma_space: float = 25

    # 多线程参数
    enable_threading: bool = True
    max_queue_size: int = 10

    # 质量控制参数
    quality_mode: str = "balanced"  # fast, balanced, quality

    def __post_init__(self):
        """参数验证和调整"""
        if self.dark_channel_size % 2 == 0:
            self.dark_channel_size += 1

        if not (0 < self.min_transmission <= 1):
            raise ValueError("最小传输图阈值必须在(0,1]范围内")
        if not (0 < self.omega <= 1):
            raise ValueError("去雾强度因子必须在(0,1]范围内")
        if not (0 <= self.temporal_weight <= 1):
            raise ValueError("时间权重必须在[0,1]范围内")

        # 根据质量模式调整参数
        if self.quality_mode == "fast":
            self.resize_factor = 0.3
            self.skip_frames = 2
            self.bilateral_d = 3
        elif self.quality_mode == "quality":
            self.resize_factor = 0.8
            self.skip_frames = 0
            self.bilateral_d = 9


class FrameBuffer:
    """帧缓冲区管理类"""

    def __init__(self, max_size: int = 5):
        """
        初始化帧缓冲区

        Args:
            max_size: 最大缓冲帧数
        """
        self.max_size = max_size
        self.frames = deque(maxlen=max_size)
        self.transmissions = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.lock = threading.Lock()

    def add_frame(self, frame: np.ndarray, transmission: np.ndarray = None):
        """
        添加新帧到缓冲区

        Args:
            frame: 输入帧
            transmission: 对应的传输图
        """
        with self.lock:
            current_time = time.time()
            self.frames.append(frame.copy())
            self.transmissions.append(transmission.copy() if transmission is not None else None)
            self.timestamps.append(current_time)

    def get_latest_frames(self, count: int = None) -> List[np.ndarray]:
        """
        获取最新的几帧

        Args:
            count: 获取的帧数，None表示获取全部

        Returns:
            List[np.ndarray]: 帧列表
        """
        with self.lock:
            if count is None:
                return list(self.frames)
            else:
                return list(self.frames)[-count:] if len(self.frames) >= count else list(self.frames)

    def get_latest_transmissions(self, count: int = None) -> List[np.ndarray]:
        """
        获取最新的几个传输图

        Args:
            count: 获取的数量

        Returns:
            List[np.ndarray]: 传输图列表
        """
        with self.lock:
            if count is None:
                return [t for t in self.transmissions if t is not None]
            else:
                recent_transmissions = list(self.transmissions)[-count:] if len(self.transmissions) >= count else list(self.transmissions)
                return [t for t in recent_transmissions if t is not None]

    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        with self.lock:
            return len(self.frames) == 0


class AtmosphericLightEstimator:
    """大气光自适应估计器"""

    def __init__(self, adaptation_rate: float = 0.1):
        """
        初始化大气光估计器

        Args:
            adaptation_rate: 自适应速率
        """
        self.adaptation_rate = adaptation_rate
        self.current_airlight = None
        self.frame_count = 0
        self.lock = threading.Lock()

    def estimate_airlight(self, image: np.ndarray, dark_channel: np.ndarray) -> np.ndarray:
        """
        估计大气光值

        Args:
            image: 输入图像
            dark_channel: 暗通道图

        Returns:
            np.ndarray: 大气光值
        """
        h, w = dark_channel.shape
        flat_image = image.reshape(-1, 3)
        flat_dark = dark_channel.reshape(-1)

        # 选择暗通道值最大的0.1%像素
        num_pixels = max(1, int(h * w * 0.001))
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]

        # 在这些像素中选择最亮的作为大气光
        brightest_pixels = flat_image[indices]
        intensities = np.sum(brightest_pixels, axis=1)
        brightest_idx = indices[np.argmax(intensities)]

        new_airlight = flat_image[brightest_idx].astype(np.float64)

        with self.lock:
            if self.current_airlight is None:
                self.current_airlight = new_airlight
            else:
                # 自适应更新
                self.current_airlight = (1 - self.adaptation_rate) * self.current_airlight + \
                                      self.adaptation_rate * new_airlight

            self.frame_count += 1
            return self.current_airlight.copy()


class RealtimeDehazingProcessor:
    """实时去雾处理器主类"""

    def __init__(self, params: Optional[RealtimeDehazingParams] = None):
        """
        初始化实时去雾处理器

        Args:
            params: 算法参数配置
        """
        self.params = params or RealtimeDehazingParams()
        self.logger = logging.getLogger(__name__)

        # 初始化组件
        self.frame_buffer = FrameBuffer(self.params.history_frames)
        self.airlight_estimator = AtmosphericLightEstimator(
            self.params.airlight_adaptation_rate
        )

        # 性能统计
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        self.fps_history = deque(maxlen=30)

        # 多线程相关
        self.processing_queue = queue.Queue(maxsize=self.params.max_queue_size)
        self.result_queue = queue.Queue(maxsize=self.params.max_queue_size)
        self.processing_thread = None
        self.is_running = False

        if self.params.enable_threading:
            self._start_processing_thread()

    def _start_processing_thread(self):
        """启动处理线程"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _processing_worker(self):
        """处理线程工作函数"""
        while self.is_running:
            try:
                # 获取待处理帧
                frame_data = self.processing_queue.get(timeout=0.1)
                if frame_data is None:  # 退出信号
                    break

                frame, frame_id = frame_data

                # 处理帧
                result = self._process_single_frame(frame)

                # 将结果放入结果队列
                if not self.result_queue.full():
                    self.result_queue.put((result, frame_id))

                self.processing_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"处理线程发生错误: {str(e)}")

    def _compute_fast_dark_channel(self, image: np.ndarray) -> np.ndarray:
        """
        快速计算暗通道

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 暗通道图
        """
        # 计算三通道最小值
        min_channel = np.min(image, axis=2)

        # 使用形态学腐蚀进行最小值滤波
        kernel_size = self.params.dark_channel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)

        return dark_channel

    def _estimate_transmission(self, image: np.ndarray,
                             atmospheric_light: np.ndarray) -> np.ndarray:
        """
        估计传输图

        Args:
            image: 输入图像
            atmospheric_light: 大气光值

        Returns:
            np.ndarray: 传输图
        """
        # 避免除零
        atmospheric_light = np.maximum(atmospheric_light, 1.0)

        # 归一化图像
        normalized = image.astype(np.float64) / atmospheric_light
        normalized_uint8 = np.clip(normalized * 255, 0, 255).astype(np.uint8)

        # 计算暗通道
        dark_norm = self._compute_fast_dark_channel(normalized_uint8)

        # 计算传输图
        transmission = 1.0 - self.params.omega * (dark_norm.astype(np.float64) / 255.0)

        return transmission

    def _refine_transmission_fast(self, transmission: np.ndarray,
                                guide_image: np.ndarray) -> np.ndarray:
        """
        快速传输图精化

        Args:
            transmission: 原始传输图
            guide_image: 引导图像

        Returns:
            np.ndarray: 精化后的传输图
        """
        # 转换为8位进行双边滤波
        transmission_8bit = (transmission * 255).astype(np.uint8)

        # 使用双边滤波进行边缘保持平滑
        refined = cv2.bilateralFilter(
            transmission_8bit,
            self.params.bilateral_d,
            self.params.bilateral_sigma_color,
            self.params.bilateral_sigma_space
        )

        return refined.astype(np.float64) / 255.0

    def _temporal_consistency_filter(self, current_transmission: np.ndarray) -> np.ndarray:
        """
        时间一致性滤波

        Args:
            current_transmission: 当前帧传输图

        Returns:
            np.ndarray: 时间一致性滤波后的传输图
        """
        if self.params.temporal_weight == 0:
            return current_transmission

        # 获取历史传输图
        history_transmissions = self.frame_buffer.get_latest_transmissions(
            self.params.history_frames - 1
        )

        if not history_transmissions:
            return current_transmission

        # 计算加权平均
        total_weight = 1.0
        filtered_transmission = current_transmission.copy()

        for i, hist_transmission in enumerate(reversed(history_transmissions)):
            if hist_transmission.shape == current_transmission.shape:
                weight = self.params.temporal_weight * (0.8 ** i)  # 指数衰减权重
                filtered_transmission += weight * hist_transmission
                total_weight += weight

        # 归一化
        filtered_transmission /= total_weight

        return filtered_transmission

    def _recover_scene_radiance(self, image: np.ndarray,
                              transmission: np.ndarray,
                              atmospheric_light: np.ndarray) -> np.ndarray:
        """
        恢复场景辐射

        Args:
            image: 输入图像
            transmission: 传输图
            atmospheric_light: 大气光值

        Returns:
            np.ndarray: 去雾后的图像
        """
        # 限制传输图最小值
        transmission = np.maximum(transmission, self.params.min_transmission)

        # 场景辐射恢复
        image_float = image.astype(np.float64)
        recovered = np.zeros_like(image_float)

        for c in range(3):
            numerator = image_float[:, :, c] - atmospheric_light[c]
            recovered[:, :, c] = numerator / transmission + atmospheric_light[c]

        # 限制到有效范围
        recovered = np.clip(recovered, 0, 255)

        return recovered.astype(np.uint8)

    def _process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧

        Args:
            frame: 输入帧

        Returns:
            np.ndarray: 处理后的帧
        """
        # 记录处理开始时间
        start_time = time.time()

        # 缩放图像以提升处理速度
        original_size = (frame.shape[1], frame.shape[0])
        if self.params.resize_factor != 1.0:
            new_size = (
                int(frame.shape[1] * self.params.resize_factor),
                int(frame.shape[0] * self.params.resize_factor)
            )
            resized_frame = cv2.resize(frame, new_size)
        else:
            resized_frame = frame

        try:
            # 1. 计算暗通道
            dark_channel = self._compute_fast_dark_channel(resized_frame)

            # 2. 估计/更新大气光（间隔更新以提升性能）
            if (self.frame_count % self.params.airlight_update_interval == 0 or
                self.airlight_estimator.current_airlight is None):
                atmospheric_light = self.airlight_estimator.estimate_airlight(
                    resized_frame, dark_channel
                )
            else:
                atmospheric_light = self.airlight_estimator.current_airlight

            # 3. 估计传输图
            transmission = self._estimate_transmission(resized_frame, atmospheric_light)

            # 4. 快速传输图精化
            transmission_refined = self._refine_transmission_fast(
                transmission, resized_frame
            )

            # 5. 时间一致性滤波
            transmission_filtered = self._temporal_consistency_filter(transmission_refined)

            # 6. 恢复场景辐射
            if self.params.resize_factor != 1.0:
                # 如果进行了缩放，需要将传输图恢复到原始尺寸
                transmission_upscaled = cv2.resize(
                    transmission_filtered, original_size, interpolation=cv2.INTER_LINEAR
                )
                recovered = self._recover_scene_radiance(
                    frame, transmission_upscaled, atmospheric_light
                )
            else:
                recovered = self._recover_scene_radiance(
                    resized_frame, transmission_filtered, atmospheric_light
                )

            # 添加到帧缓冲区
            self.frame_buffer.add_frame(frame, transmission_refined)

            # 记录处理时间
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            return recovered

        except Exception as e:
            self.logger.error(f"处理帧时发生错误: {str(e)}")
            return frame  # 返回原始帧

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        处理输入帧（外部接口）

        Args:
            frame: 输入帧

        Returns:
            Optional[np.ndarray]: 处理后的帧，如果使用多线程可能返回None
        """
        if frame is None or frame.size == 0:
            return None

        self.frame_count += 1

        # 跳帧处理
        if self.params.skip_frames > 0 and self.frame_count % (self.params.skip_frames + 1) != 0:
            return frame  # 返回原始帧

        if self.params.enable_threading:
            # 多线程模式
            try:
                # 将帧加入处理队列
                if not self.processing_queue.full():
                    self.processing_queue.put((frame, self.frame_count))

                # 尝试获取处理结果
                try:
                    result, frame_id = self.result_queue.get_nowait()
                    return result
                except queue.Empty:
                    return frame  # 暂无结果，返回原始帧

            except Exception as e:
                self.logger.error(f"多线程处理错误: {str(e)}")
                return frame
        else:
            # 单线程模式
            return self._process_single_frame(frame)

    def get_performance_stats(self) -> dict:
        """
        获取性能统计信息

        Returns:
            dict: 性能统计数据
        """
        if not self.processing_times:
            return {}

        avg_time = np.mean(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            'avg_processing_time': avg_time,
            'fps': fps,
            'frame_count': self.frame_count,
            'queue_sizes': {
                'processing': self.processing_queue.qsize() if self.params.enable_threading else 0,
                'result': self.result_queue.qsize() if self.params.enable_threading else 0
            },
            'buffer_size': len(self.frame_buffer.frames)
        }

    def cleanup(self):
        """清理资源"""
        if self.params.enable_threading and self.is_running:
            self.is_running = False

            # 发送退出信号
            try:
                self.processing_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

            # 等待处理线程结束
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)


class VideoDehazer:
    """视频去雾器"""

    def __init__(self, params: Optional[RealtimeDehazingParams] = None):
        """
        初始化视频去雾器

        Args:
            params: 去雾参数
        """
        self.params = params or RealtimeDehazingParams()
        self.processor = RealtimeDehazingProcessor(self.params)
        self.logger = logging.getLogger(__name__)

    def process_video_file(self, input_path: str, output_path: str,
                          progress_callback: Optional[Callable] = None) -> bool:
        """
        处理视频文件

        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            progress_callback: 进度回调函数

        Returns:
            bool: 处理是否成功
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.logger.error(f"无法打开视频文件: {input_path}")
            return False

        try:
            # 获取视频属性
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_count = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                processed_frame = self.processor.process_frame(frame)
                if processed_frame is not None:
                    out.write(processed_frame)
                else:
                    out.write(frame)  # 写入原始帧

                frame_count += 1

                # 进度回调
                if progress_callback and frame_count % 30 == 0:
                    progress = frame_count / total_frames if total_frames > 0 else 0
                    progress_callback(progress)

                # 性能监控
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed
                    self.logger.info(f"处理进度: {frame_count}/{total_frames}, "
                                   f"当前FPS: {current_fps:.1f}")

            # 清理资源
            cap.release()
            out.release()

            total_time = time.time() - start_time
            avg_fps = frame_count / total_time
            self.logger.info(f"视频处理完成: {frame_count}帧, "
                           f"总耗时: {total_time:.1f}秒, "
                           f"平均FPS: {avg_fps:.1f}")

            return True

        except Exception as e:
            self.logger.error(f"视频处理过程中发生错误: {str(e)}")
            return False
        finally:
            self.processor.cleanup()

    def process_camera_stream(self, camera_id: int = 0,
                            display_result: bool = True) -> None:
        """
        处理摄像头实时流

        Args:
            camera_id: 摄像头ID
            display_result: 是否显示结果
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.logger.error(f"无法打开摄像头: {camera_id}")
            return

        try:
            # 设置摄像头参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            self.logger.info("开始实时去雾处理，按'q'键退出")

            fps_calculator = FPSCalculator()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                start_time = time.time()
                processed_frame = self.processor.process_frame(frame)
                process_time = time.time() - start_time

                if processed_frame is None:
                    processed_frame = frame

                # 计算FPS
                fps = fps_calculator.update()

                if display_result:
                    # 在图像上显示性能信息
                    info_text = f"FPS: {fps:.1f}, Process: {process_time*1000:.1f}ms"
                    cv2.putText(processed_frame, info_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 显示结果
                    cv2.imshow('Realtime Dehazing', processed_frame)
                    cv2.imshow('Original', frame)

                    # 检查退出键
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except Exception as e:
            self.logger.error(f"实时处理过程中发生错误: {str(e)}")
        finally:
            cap.release()
            if display_result:
                cv2.destroyAllWindows()
            self.processor.cleanup()


class FPSCalculator:
    """FPS计算器"""

    def __init__(self, window_size: int = 30):
        """
        初始化FPS计算器

        Args:
            window_size: 计算窗口大小
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)

    def update(self) -> float:
        """
        更新并计算FPS

        Returns:
            float: 当前FPS
        """
        current_time = time.time()
        self.timestamps.append(current_time)

        if len(self.timestamps) < 2:
            return 0.0

        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span > 0:
            return (len(self.timestamps) - 1) / time_span
        else:
            return 0.0


def create_realtime_dehazer(quality: str = "balanced") -> VideoDehazer:
    """
    创建实时去雾器的便捷函数

    Args:
        quality: 质量级别 ("fast", "balanced", "quality")

    Returns:
        VideoDehazer: 配置好的视频去雾器
    """
    params = RealtimeDehazingParams(quality_mode=quality)
    return VideoDehazer(params)


def main():
    """主函数：演示实时去雾算法的使用"""

    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 创建去雾器
    dehazer = create_realtime_dehazer("balanced")

    print("实时去雾演示程序")
    print("1. 处理摄像头实时流")
    print("2. 处理视频文件")

    choice = input("请选择模式 (1/2): ").strip()

    try:
        if choice == "1":
            # 实时摄像头处理
            print("开始摄像头实时去雾...")
            dehazer.process_camera_stream(camera_id=0, display_result=True)

        elif choice == "2":
            # 视频文件处理
            input_path = input("请输入视频文件路径: ").strip()
            output_path = input("请输入输出文件路径: ").strip()

            if not output_path:
                output_path = "dehazed_output.mp4"

            def progress_callback(progress):
                print(f"处理进度: {progress*100:.1f}%")

            print("开始处理视频文件...")
            success = dehazer.process_video_file(
                input_path, output_path, progress_callback
            )

            if success:
                print(f"视频处理完成，输出文件: {output_path}")
            else:
                print("视频处理失败")

        else:
            print("无效选择")

    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()