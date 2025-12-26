#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, List, Optional


class LaneDetectionConfig:
    CANNY_LOW_THRESH = 50
    CANNY_HIGH_THRESH = 150
    GAUSSIAN_KERNEL = (5, 5)
    HOUGH_RHO = 1
    HOUGH_THETA = np.pi / 180
    HOUGH_THRESHOLD = 15      
    HOUGH_MIN_LINE_LEN = 30   
    HOUGH_MAX_LINE_GAP = 15   
    SLOPE_THRESHOLD = 0.2     
    
    COLOR_LEFT_LANE = (0, 0, 255)    
    COLOR_RIGHT_LANE = (255, 0, 0)   
    COLOR_MASK = (0, 255, 0)         
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2
    RESULT_SUFFIX = "_lane_detected.mp4"


class LaneDetector:
    def __init__(self, config: LaneDetectionConfig = None):
        self.config = config or LaneDetectionConfig()
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.config.GAUSSIAN_KERNEL, 0)
        edges = cv2.Canny(blur, self.config.CANNY_LOW_THRESH, self.config.CANNY_HIGH_THRESH)
        return edges
    
    def _create_roi_mask(self, edges: np.ndarray) -> np.ndarray:
        h, w = edges.shape[:2]
        roi_vertices = np.array([[(0, h), (w//2, h//2), (w, h)]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges
    
    def _classify_lane_lines(self, lines: Optional[np.ndarray], frame_width: int) -> Tuple[List, List]:
        left_lines = []
        right_lines = []
        if lines is None or len(lines) == 0:
            return left_lines, right_lines
        
        mid_x = frame_width / 2
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            if slope < -self.config.SLOPE_THRESHOLD and (x1 < mid_x or x2 < mid_x):
                left_lines.append((x1, y1, x2, y2))
            elif slope > self.config.SLOPE_THRESHOLD and (x1 > mid_x or x2 > mid_x):
                right_lines.append((x1, y1, x2, y2))
        return left_lines, right_lines
    
    def detect_and_visualize(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        edges = self._preprocess_frame(frame)
        masked_edges = self._create_roi_mask(edges)
        lines = cv2.HoughLinesP(
            masked_edges,
            self.config.HOUGH_RHO,
            self.config.HOUGH_THETA,
            self.config.HOUGH_THRESHOLD,
            minLineLength=self.config.HOUGH_MIN_LINE_LEN,
            maxLineGap=self.config.HOUGH_MAX_LINE_GAP
        )
        left_lines, right_lines = self._classify_lane_lines(lines, w)
        
        for x1, y1, x2, y2 in left_lines:
            cv2.line(frame_copy, (x1, y1), (x2, y2), self.config.COLOR_LEFT_LANE, 4)
        for x1, y1, x2, y2 in right_lines:
            cv2.line(frame_copy, (x1, y1), (x2, y2), self.config.COLOR_RIGHT_LANE, 4)
        
        if left_lines and right_lines:
            left_bottom = max(left_lines, key=lambda l: l[3])[:2]
            left_top = min(left_lines, key=lambda l: l[3])[:2]
            right_bottom = max(right_lines, key=lambda l: l[3])[:2]
            right_top = min(right_lines, key=lambda l: l[3])[:2]
            
            mask_pts = np.array([left_bottom, left_top, right_top, right_bottom], dtype=np.int32)
            mask_layer = frame_copy.copy()
            cv2.fillPoly(mask_layer, [mask_pts], self.config.COLOR_MASK)
            cv2.addWeighted(mask_layer, 0.2, frame_copy, 0.8, 0, frame_copy)
        
        cv2.putText(frame_copy, "红=左车道 蓝=右车道", (10, h-20), 
                    self.config.FONT, self.config.FONT_SCALE, (255,255,255), self.config.FONT_THICKNESS)
        
        return frame_copy, len(left_lines), len(right_lines)


class VideoProcessor:
    def __init__(self, detector: LaneDetector, config: LaneDetectionConfig = None):
        self.detector = detector
        self.config = config or LaneDetectionConfig()
    
    def _create_split_screen(self, original: np.ndarray, detected: np.ndarray) -> np.ndarray:
        h, w = original.shape[:2]
        detected = cv2.resize(detected, (w, h), interpolation=cv2.INTER_LINEAR)
        split_frame = np.hstack((original, detected))
        
        cv2.putText(split_frame, "Original", (20, 30),
                    self.config.FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(split_frame, "Lane Detection", (w + 20, 30),
                    self.config.FONT, 0.8, (255, 255, 255), 2)
        return split_frame
    
    def process_video(self, video_path: str) -> None:
        max_frames = 200
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        frame_delay = int(1000 / fps) if fps > 0 else 33
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or max_frames), max_frames)
        
        result_path = video_path.parent / f"{video_path.stem}{self.config.RESULT_SUFFIX}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(result_path), fourcc, fps, (w * 2, h))
        
        paused = False
        step_mode = False
        frame_idx = 0
        
        print("=" * 50)
        print("车道线检测（分屏版-最终版）")
        print("=" * 50)
        print("操作：空格单步 P=暂停/继续 Q=退出")
        print(f"适配帧率：{fps} FPS | 最大处理帧数：{max_frames}")
        print("=" * 50)
        
        cv2.namedWindow("Lane Detection (Split Screen)", cv2.WINDOW_NORMAL)
        max_win_width = 1920
        if w * 2 > max_win_width:
            new_win_w = max_win_width
            new_win_h = int(h * (max_win_width / (w * 2)))
        else:
            new_win_w = w * 2
            new_win_h = h
        cv2.resizeWindow("Lane Detection (Split Screen)", new_win_w, new_win_h)
        
        try:
            while cap.isOpened() and frame_idx < total_frames:
                if step_mode:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord(' '):
                        step_mode = False
                    elif key == ord('q'):
                        break
                    continue
                
                if not paused:
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        frame_idx += 1
                        continue
                    
                    detected_frame, left_count, right_count = self.detector.detect_and_visualize(frame)
                    split_frame = self._create_split_screen(frame, detected_frame)
                    
                    writer.write(split_frame)
                    frame_idx += 1
                
                cv2.imshow("Lane Detection (Split Screen)", split_frame)
                
                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                elif key == ord(' '):
                    step_mode = True
        
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            print(f"\n处理完成！结果保存至：{result_path}")
            print(f"实际处理帧数：{frame_idx}")


def main():
    parser = argparse.ArgumentParser(description="车道线检测（分屏版-最终版）")
    parser.add_argument("video_path", type=str, help="视频路径（如：road_video_fixed.mp4）")
    args = parser.parse_args()
    
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("错误：请安装依赖 → pip install opencv-python numpy")
        return
    
    config = LaneDetectionConfig()
    detector = LaneDetector(config)
    processor = VideoProcessor(detector, config)
    
    try:
        processor.process_video(args.video_path)
    except Exception as e:
        print(f"处理失败：{str(e)}")


if __name__ == "__main__":
    main()
