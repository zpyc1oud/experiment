"""
FocusVision - 课堂专注度检测系统的Web应用

该模块实现了FocusVision系统的Web界面和后端处理逻辑，基于Flask框架。
主要功能包括：
1. 提供直观的Web界面，支持文件上传和流媒体处理
2. 实时处理和分析视频中的专注度数据
3. 显示处理进度和预览图像
4. 生成可视化的结果报告
5. 管理文件上传和结果输出

该应用支持多种视频输入方式：
- 本地视频文件上传
- 摄像头实时处理
- RTSP流处理
- IP摄像头连接

版本：1.0.0
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, Response
from werkzeug.utils import secure_filename
import cv2
import time
import pandas as pd
from pathlib import Path
import threading
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import datetime
from focus_vision.focus_detector import FocusDetector
from focus_vision.utils import plot_attention_over_time
from collections import deque

# 配置日志系统
def setup_logger():
    """
    配置日志系统，创建文件和控制台处理器
    
    设置一个功能完善的日志系统，包含以下特性：
    1. 同时输出到控制台和文件
    2. 日志文件按日期命名，便于查找
    3. 自动滚动日志文件，避免文件过大
    4. 不同级别的日志使用不同颜色和格式
    
    Returns:
        logging.Logger: 配置好的日志记录器对象
    """
    # 创建logs目录
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 日志文件名包含日期
    log_filename = log_dir / f"focus_vision_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 创建Logger对象
    logger = logging.getLogger('FocusVision')
    logger.setLevel(logging.DEBUG)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 文件处理器 - 使用RotatingFileHandler限制日志文件大小
    file_handler = RotatingFileHandler(
        log_filename, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 设置日志格式
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 初始化日志记录器
logger = setup_logger()
logger.info("FocusVision应用启动")

# 检查GPU可用性并进行配置
def setup_gpu():
    """
    检查GPU可用性并进行配置
    
    该函数实现：
    1. 检测系统中可用的GPU设备
    2. 为TensorFlow和PyTorch配置GPU使用策略
    3. 配置TensorFlow的显存动态分配，避免占用全部GPU内存
    4. 记录GPU信息到日志
    
    Returns:
        bool: GPU是否可用并成功配置
    """
    gpu_available = False
    
    # 检查TensorFlow (用于DeepFace)
    try:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            logger.info(f"找到 {len(physical_devices)} 个GPU设备，正在配置TensorFlow...")
            # 允许TensorFlow动态分配显存
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info("TensorFlow GPU配置完成!")
            gpu_available = True
        else:
            logger.warning("未检测到可用于TensorFlow的GPU设备")
    except Exception as e:
        logger.error(f"配置TensorFlow GPU时出错: {e}")
    
    # 检查PyTorch (用于YOLO)
    try:
        import torch
        if torch.cuda.is_available():
            torch_gpu_available = True
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info(f"PyTorch可以使用GPU: {device_name} (共{device_count}个设备)")
            gpu_available = True
        else:
            logger.warning("未检测到可用于PyTorch的GPU设备")
    except Exception as e:
        logger.error(f"检测PyTorch GPU时出错: {e}")
    
    return gpu_available

# 在应用启动时检测和配置GPU
GPU_AVAILABLE = setup_gpu()

app = Flask(__name__) # 创建Flask应用

# --- 配置 --- #
UPLOAD_FOLDER = os.path.join('storage', 'uploads')
OUTPUT_FOLDER = os.path.join('storage', 'output_data') # 用于存储处理后的视频和CSV
TEMP_FOLDER = os.path.join('storage', 'temp_data') # 用于存储处理过程中的临时文件
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 500  # 500 MB 上传限制

def configure_app():
    """
    配置Flask应用，确保使用绝对路径
    
    该函数实现：
    1. 将相对路径转换为绝对路径，避免路径错误
    2. 确保所有必要的目录存在
    3. 更新Flask应用的配置
    4. 记录配置信息到日志
    """
    # 设置绝对路径
    global OUTPUT_FOLDER, UPLOAD_FOLDER, TEMP_FOLDER
    
    # 检查是否使用相对路径，如果是则转换为绝对路径
    if not os.path.isabs(OUTPUT_FOLDER):
        OUTPUT_FOLDER = os.path.abspath(OUTPUT_FOLDER)
    if not os.path.isabs(UPLOAD_FOLDER):
        UPLOAD_FOLDER = os.path.abspath(UPLOAD_FOLDER)
    if not os.path.isabs(TEMP_FOLDER):
        TEMP_FOLDER = os.path.abspath(TEMP_FOLDER)
    
    # 确保目录存在
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(TEMP_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # 更新应用配置
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
    app.config['TEMP_FOLDER'] = TEMP_FOLDER
    
    logger.info(f"配置完成 - 输出目录: {OUTPUT_FOLDER}")
    logger.info(f"配置完成 - 上传目录: {UPLOAD_FOLDER}")
    logger.info(f"配置完成 - 临时目录: {TEMP_FOLDER}")

# 调用配置函数确保使用绝对路径
configure_app()

# 确保上传和输出文件夹存在
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
Path(TEMP_FOLDER).mkdir(parents=True, exist_ok=True)

# 创建一个占位图片，用于processing页面初始加载
placeholder_img_path = os.path.join('templates', 'placeholder.jpg')
if not os.path.exists(placeholder_img_path):
    placeholder = np.ones((300, 400, 3), dtype=np.uint8) * 255  # 白色背景
    cv2.putText(placeholder, "Waiting...", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite(placeholder_img_path, placeholder)

# 全局变量用于跟踪处理状态
processing_status = {
    'status': 'idle',  # idle, running, paused, completed
    'progress': 0,
    'current_frame': 0,
    'total_frames': 0,
    'elapsed_time': 0,
    'focus_data': [],
    'average_focus': 0,
    'last_frame_path': None,
    'processing_thread': None,
    'video_path': None,
    'output_video_path': None,
    'output_csv_path': None,
    'output_plot_path': None,
    'is_paused': False,
    'is_stream': False,  # 新增：标识当前是否为流模式
    'stream_source': None,  # 新增：存储流来源信息
    'stream_type': None,  # 新增：存储流类型
    'stream_description': None,  # 新增：存储流描述
    'gpu_enabled': GPU_AVAILABLE,  # 新增：标记是否启用了GPU
    'processing_fps': 0.0,  # 新增：当前处理帧率
    'avg_processing_time': 0.0  # 新增：平均每帧处理时间（毫秒）
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 初始化专注度检测器 (可以稍后根据需要加载不同模型)
# 如果GPU可用，则使用GPU，并启用并行模型池
pool_size = 4 if GPU_AVAILABLE else max(1, min(4, (os.cpu_count() or 2)))
logger.info(f"初始化专注度检测器，GPU可用: {GPU_AVAILABLE}，模型池大小: {pool_size}")
# DeepFace并发数：GPU下建议小于等于4，CPU下可与模型池一致或略小
deepface_cc = 4 if GPU_AVAILABLE else max(1, min(pool_size, 4))
focus_detector = FocusDetector(use_gpu=GPU_AVAILABLE, model_pool_size=pool_size, deepface_concurrency=deepface_cc)

@app.route('/')
def index():
    """
    首页路由处理函数
    
    提供系统主页面，显示文件上传和视频流处理选项。
    同时将GPU可用状态传递给模板，以便在界面上显示。
    
    Returns:
        str: 渲染后的HTML模板
    """
    # 传递GPU状态到模板
    return render_template('index.html', gpu_available=GPU_AVAILABLE)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    处理视频文件上传的路由函数
    
    实现以下功能：
    1. 接收并验证上传的视频文件
    2. 安全地保存文件到指定目录
    3. 设置处理参数和输出路径
    4. 启动异步处理线程
    5. 重定向到处理状态页面
    
    Returns:
        Response: 重定向到处理页面的响应
    """
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        # 准备处理文件
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_video_filename = f"{Path(filename).stem}_processed_{timestamp}.mp4"
        output_csv_filename = f"{Path(filename).stem}_data_{timestamp}.csv"
        output_plot_filename = f"{Path(filename).stem}_attention_plot_{timestamp}.png"

        output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], output_video_filename)
        output_csv_path = os.path.join(app.config['OUTPUT_FOLDER'], output_csv_filename)
        output_plot_path = os.path.join(app.config['OUTPUT_FOLDER'], output_plot_filename)
        
        # 更新处理状态
        global processing_status
        processing_status = {
            'status': 'running',
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,
            'elapsed_time': 0,
            'focus_data': [],
            'average_focus': 0,
            'last_frame_path': None,
            'processing_thread': None,
            'video_path': video_path,
            'output_video_path': output_video_path,
            'output_csv_path': output_csv_path,
            'output_plot_path': output_plot_path,
            'is_paused': False,
            'is_stream': False,
            'stream_source': None,
            'stream_type': None,
            'stream_description': None,
            'gpu_enabled': GPU_AVAILABLE, 
            'processing_fps': 0.0,
            'avg_processing_time': 0.0
        }
        
        # 启动异步处理线程
        processing_thread = threading.Thread(
            target=process_video_async,
            args=(video_path, output_video_path, output_csv_path, output_plot_path)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        processing_status['processing_thread'] = processing_thread
        
        # 重定向到处理页面
        return redirect(url_for('processing_view'))
    
    return redirect(url_for('index'))

# 新增：处理流模式视频
@app.route('/stream', methods=['POST'])
def handle_stream():
    """处理视频流输入，支持本地摄像头、RTSP、HTTP流和IP摄像头"""
    try:
        stream_type = request.form.get('stream_type')
        stream_source = None
        stream_description = None
        
        if stream_type == 'webcam':
            # 本地摄像头
            webcam_id = int(request.form.get('webcam_id', 0))
            stream_source = webcam_id
            stream_description = f"摄像头 #{webcam_id}"
        
        elif stream_type == 'rtsp':
            # RTSP流
            rtsp_url = request.form.get('rtsp_url')
            if not rtsp_url:
                return render_template('index.html', error="RTSP流地址不能为空")
            stream_source = rtsp_url
            stream_description = f"RTSP流: {rtsp_url}"
        
        elif stream_type == 'http':
            # HTTP流
            http_url = request.form.get('http_url')
            if not http_url:
                return render_template('index.html', error="HTTP流地址不能为空")
            stream_source = http_url
            stream_description = f"HTTP流: {http_url}"
        
        elif stream_type == 'ip_camera':
            # IP摄像头
            ip_address = request.form.get('ip_address')
            ip_port = request.form.get('ip_port', '8080')
            ip_username = request.form.get('ip_username', '')
            ip_password = request.form.get('ip_password', '')
            
            if not ip_address:
                return render_template('index.html', error="IP摄像头地址不能为空")
            
            # 构建IP摄像头URL
            if ip_username and ip_password:
                ip_url = f"http://{ip_username}:{ip_password}@{ip_address}:{ip_port}/video"
            else:
                ip_url = f"http://{ip_address}:{ip_port}/video"
            
            stream_source = ip_url
            stream_description = f"IP摄像头: {ip_address}:{ip_port}"
        
        else:
            return render_template('index.html', error="未知的流类型")
        
        # 准备输出文件
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_base_name = f"stream_{stream_type}_{timestamp}"
        output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{output_base_name}.mp4")
        output_csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{output_base_name}_data.csv")
        output_plot_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{output_base_name}_plot.png")
        
        # 更新处理状态
        global processing_status
        processing_status = {
            'status': 'running',
            'progress': 0,
            'current_frame': 0,
            'total_frames': 0,  # 流模式没有总帧数
            'elapsed_time': 0,
            'focus_data': [],
            'average_focus': 0,
            'last_frame_path': None,
            'processing_thread': None,
            'video_path': None,
            'output_video_path': output_video_path,
            'output_csv_path': output_csv_path,
            'output_plot_path': output_plot_path,
            'is_paused': False,
            'is_stream': True,
            'stream_source': stream_source,
            'stream_description': stream_description,
            'stream_type': stream_type,
            'gpu_enabled': GPU_AVAILABLE,
            'processing_fps': 0.0,
            'avg_processing_time': 0.0
        }
        
        # 启动异步处理线程
        processing_thread = threading.Thread(
            target=process_stream_async,
            args=(stream_source, stream_type, output_video_path, output_csv_path, output_plot_path)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        processing_status['processing_thread'] = processing_thread
        
        # 重定向到处理页面
        return redirect(url_for('processing_view'))
        
    except Exception as e:
        return render_template('index.html', error=f"处理流视频时出错: {str(e)}")

# 新增：实时流处理函数
def process_stream_async(stream_source, stream_type, output_video_path, output_csv_path, output_plot_path):
    """异步处理视频流，并定期更新处理状态"""
    global processing_status
    
    try:
        # 准备临时文件夹
        temp_frames_dir = os.path.join(app.config['TEMP_FOLDER'], f"stream_{int(time.time())}")
        Path(temp_frames_dir).mkdir(parents=True, exist_ok=True)
        logger.debug(f"创建临时目录: {temp_frames_dir}")
        
        # 打开视频流
        if stream_type == 'webcam':
            # 对于摄像头，stream_source是一个整数ID
            logger.info(f"正在打开本地摄像头: ID={stream_source}")
            cap = cv2.VideoCapture(int(stream_source))
        else:
            # 对于其他流，stream_source是URL字符串
            logger.info(f"正在连接流: {stream_source}")
            cap = cv2.VideoCapture(stream_source)
        
        if not cap.isOpened():
            error_msg = f"无法打开视频流: {stream_source}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # 某些流无法获取FPS
            fps = 30  # 使用默认值
        logger.info(f"视频流属性: 宽度={width}, 高度={height}, FPS={fps}")

        # 设置输出视频
        try:
            # 为避免libopenh264相关错误，统一使用mp4v编码器写入MP4容器
            # mp4v在Windows + OpenCV默认环境兼容性较好且不依赖外部H.264库
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # 检查视频写入器是否正确创建
            if not out.isOpened():
                error_msg = "无法创建视频写入器，可能是编码器问题"
                logger.error(error_msg)
                raise Exception(error_msg)
        except Exception as e:
            logger.error(f"创建视频写入器失败: {e}")
            cap.release()
            return
        
        # 重置检测器状态
        focus_detector.focus_data = []
        focus_detector.frame_count = 0
        
        # 记录开始时间
        start_time = time.time()
        frame_count = 0
        written_count = 0
        focus_data_list = []
        max_frames_to_keep = 100  # 最多保留的实时数据点数
        processing_times = []  # 用于存储每帧处理时间
        inflight = deque()
        max_inflight = max(1, focus_detector.model_pool_size * 2)
        
        # 主处理循环
        while True:
            # 检查是否暂停
            while processing_status['is_paused']:
                time.sleep(0.5)
                continue
            
            # 检查是否需要停止
            if processing_status['status'] == 'idle':
                break
            
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                # 对于流，尝试重新连接
                logger.warning("无法读取帧，尝试重新连接...")
                time.sleep(1)
                cap.release()
                
                if stream_type == 'webcam':
                    cap = cv2.VideoCapture(int(stream_source))
                else:
                    cap = cv2.VideoCapture(stream_source)
                
                if not cap.isOpened():
                    logger.error("重新连接失败，结束流处理")
                    break
                
                continue
            
            # 提交当前帧进行并行处理
            submit_time = time.time()
            future = focus_detector.process_frame_async(frame)
            inflight.append((frame_count, submit_time, future))

            # 当在途任务超限时，取出最早结果并写入
            if len(inflight) > max_inflight:
                idx, st, fut = inflight.popleft()
                vis_frame, frame_data = fut.result()
                frame_processing_time = time.time() - st
                processing_times.append(frame_processing_time)

                # 计算经过时间
                elapsed_time = time.time() - start_time

                # 计算平均专注度
                avg_focus = 0
                if frame_data['students']:
                    avg_focus = sum(student['focus_score'] for student in frame_data['students']) / len(frame_data['students'])

                # 添加到实时数据
                focus_data_list.append({
                    'frame': idx,
                    'focus': avg_focus
                })

                # 只保留最近的N个数据点
                if len(focus_data_list) > max_frames_to_keep:
                    focus_data_list = focus_data_list[-max_frames_to_keep:]

                # 保存当前帧到临时文件，用于流式显示
                current_frame_path = os.path.join(temp_frames_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(current_frame_path, vis_frame)

                # 写入视频
                out.write(vis_frame)

                # 计算性能统计
                avg_processing_time = sum(processing_times[-30:]) / min(len(processing_times), 30) if processing_times else 0
                current_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

                # 更新处理状态 - 使用绝对路径
                processing_status['current_frame'] = idx
                processing_status['elapsed_time'] = elapsed_time
                processing_status['last_frame_path'] = os.path.abspath(current_frame_path)
                processing_status['focus_data'] = focus_data_list
                processing_status['average_focus'] = sum(item['focus'] for item in focus_data_list) / len(focus_data_list) if focus_data_list else 0
                processing_status['processing_fps'] = current_fps
                processing_status['avg_processing_time'] = avg_processing_time * 1000  # 转换为毫秒

                # 添加到CSV数据（以写入索引为准）
                if written_count % 5 == 0:
                    focus_detector._append_csv_data(frame_data, idx)

                written_count += 1
            
            frame_count += 1
            
            # 每10帧打印一次状态
            if frame_count % 10 == 0:
                avg_time_ms = sum(processing_times[-30:]) / min(len(processing_times), 30) * 1000 if processing_times else 0
                avg_fps = 1.0 / (sum(processing_times[-30:]) / min(len(processing_times), 30)) if processing_times and sum(processing_times[-30:]) > 0 else 0
                logger.info(f"流处理中: 已处理{frame_count}帧，运行时间 {elapsed_time:.1f}秒 | FPS: {avg_fps:.2f} | 处理时间: {avg_time_ms:.1f}ms/帧")
            
            # 控制帧率，避免CPU占用过高（使用最近一次处理时间）
            last_time = processing_times[-1] if processing_times else 0
            if not GPU_AVAILABLE and last_time < 0.03:
                time.sleep(0.01)
        
        # 完成处理
        cap.release()
        out.release()

        # 刷新剩余在途任务
        while inflight:
            idx, st, fut = inflight.popleft()
            vis_frame, frame_data = fut.result()
            current_frame_path = os.path.join(temp_frames_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(current_frame_path, vis_frame)
            out.write(vis_frame)
            processing_times.append(time.time() - st)
            written_count += 1
            processing_status['current_frame'] = idx
            processing_status['last_frame_path'] = os.path.abspath(current_frame_path)
            if written_count % 5 == 0:
                focus_detector._append_csv_data(frame_data, idx)
        
        # 保存CSV数据和生成曲线图
        if focus_detector.focus_data:
            focus_detector._save_csv(output_csv_path)
            
            # 从CSV生成注意力曲线图
            if os.path.exists(output_csv_path):
                try:
                    attention_df = pd.read_csv(output_csv_path)
                    if not attention_df.empty:
                        # 确保必要的列存在，否则创建一个基本的CSV
                        required_columns = ['frame', 'focus_score']
                        if not all(col in attention_df.columns for col in required_columns):
                            # 如果缺少必要的列，创建一个基本的CSV
                            basic_data = []
                            for idx, item in enumerate(focus_data_list):
                                basic_data.append({
                                    'frame': item['frame'],
                                    'focus_score': item['focus'],
                                    'student_id': 'all'
                                })
                            attention_df = pd.DataFrame(basic_data)
                            attention_df.to_csv(output_csv_path, index=False)
                            print(f"创建了基本的CSV数据: {output_csv_path}")
                        
                        # 生成曲线图
                        if 'student_id' in attention_df.columns and 'focus_score' in attention_df.columns and 'frame' in attention_df.columns:
                            agg_df = attention_df.groupby('frame')['focus_score'].mean().reset_index()
                            plot_attention_over_time(agg_df, output_plot_path, student_id='Average Attention')
                        else:
                            # 尝试使用基本列生成图表
                            plot_attention_over_time(attention_df, output_plot_path)
                except Exception as e:
                    print(f"处理CSV和生成图表时出错: {e}")
        
        # 标记处理完成，并确保进度显示为100%
        processing_status['progress'] = 100
        processing_status['status'] = 'completed'
        
        # 更新文件路径为绝对路径
        processing_status['output_video_path'] = os.path.abspath(output_video_path)
        processing_status['output_csv_path'] = os.path.abspath(output_csv_path)
        processing_status['output_plot_path'] = os.path.abspath(output_plot_path)
        
        # 计算统计信息并转换为results.html期望的格式
        raw_stats = focus_detector._calculate_statistics()
        
        # 为results.html页面准备统计信息
        processing_stats = {
            'total_frames_processed': frame_count,
            'average_focus_all': raw_stats.get('overall_avg_focus', 0.0),
            'processing_time_seconds': elapsed_time,
            'average_fps': 1.0 / (sum(processing_times) / len(processing_times)) if processing_times and sum(processing_times) > 0 else 0,
            'average_processing_time_ms': sum(processing_times) / len(processing_times) * 1000 if processing_times else 0,
            'gpu_enabled': GPU_AVAILABLE
        }
        
        # 构建学生摘要信息
        students_summary = []
        student_avg_focus = raw_stats.get('student_avg_focus', {})
        student_emotions = raw_stats.get('student_dominant_emotions', {})
        
        for student_id, avg_focus in student_avg_focus.items():
            students_summary.append({
                'id': student_id,
                'average_focus': avg_focus,
                'dominant_emotion': student_emotions.get(student_id, 'unknown')
            })
        
        if students_summary:
            processing_stats['students_summary'] = students_summary
            
        # 将统计信息保存到processing_status，以便results路由可以使用
        processing_status['processing_stats'] = processing_stats
        
    except Exception as e:
        logger.error(f"流处理过程中出错: {e}")
        processing_status['status'] = 'error'
        processing_status['error'] = str(e)

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """停止当前正在进行的流处理"""
    global processing_status
    
    if processing_status['is_stream'] and processing_status['status'] in ['running', 'paused']:
        processing_status['status'] = 'idle'  # 设置为idle会触发流处理线程退出
        
        # 给流处理线程一些时间来清理
        time.sleep(1)
        
        return jsonify({'status': 'stopped', 'message': '流处理已停止'})
    else:
        return jsonify({'status': 'error', 'message': '当前没有运行中的流处理'})

@app.route('/processing')
def processing_view():
    # 显示处理状态页面
    return render_template('processing.html')

@app.route('/processing_status')
def get_processing_status():
    # 返回当前处理状态的JSON
    global processing_status
    status_copy = processing_status.copy()
    
    # 移除不能JSON序列化的对象
    if 'processing_thread' in status_copy:
        del status_copy['processing_thread']
    
    return jsonify(status_copy)

@app.route('/pause_processing', methods=['POST'])
def pause_processing():
    global processing_status
    processing_status['is_paused'] = True
    processing_status['status'] = 'paused'
    return jsonify({'status': 'paused'})

@app.route('/resume_processing', methods=['POST'])
def resume_processing():
    global processing_status
    processing_status['is_paused'] = False
    processing_status['status'] = 'running'
    return jsonify({'status': 'running'})

@app.route('/stream_video')
def stream_video():
    # 实时流式处理后的视频帧
    def generate_frames():
        global processing_status
        last_frame_time = 0
        
        while processing_status['status'] in ['running', 'paused']:
            if processing_status['last_frame_path'] and os.path.exists(processing_status['last_frame_path']):
                # 每隔一段时间读取最新帧
                current_time = time.time()
                if current_time - last_frame_time > 0.1:  # 控制帧率，不要太快
                    last_frame_time = current_time
                    frame = cv2.imread(processing_status['last_frame_path'])
                    if frame is not None:
                        # 添加处理进度信息
                        progress = processing_status['progress']
                        text = f"Progress: {progress:.1f}% - Frame: {processing_status['current_frame']}/{processing_status['total_frames']}"
                        
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), 2)
                        
                        # 转换为JPEG用于流式传输
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)  # 小暂停，避免CPU占用过高
        
        # 处理完成后返回最终帧
        if processing_status['status'] == 'completed' and processing_status['last_frame_path']:
            frame = cv2.imread(processing_status['last_frame_path'])
            if frame is not None:
                cv2.putText(frame, "Processing completed! 100%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def process_video_async(video_path, output_video_path, output_csv_path, output_plot_path):
    """异步处理视频，并定期更新处理状态"""
    global processing_status
    
    try:
        # 准备临时文件夹，用于存储处理过程中的帧
        temp_frames_dir = os.path.join(app.config['TEMP_FOLDER'], f"frames_{int(time.time())}")
        Path(temp_frames_dir).mkdir(parents=True, exist_ok=True)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 更新总帧数
        processing_status['total_frames'] = total_frames
        
        # 设置输出视频
        try:
            # 为避免libopenh264相关错误，统一使用mp4v编码器写入MP4容器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # 检查视频写入器是否正确创建
            if not out.isOpened():
                raise Exception("无法创建视频写入器，可能是编码器问题")
        except Exception as e:
            logger.error(f"创建视频写入器失败: {e}")
            cap.release()
            return
        
        # 重置检测器状态
        focus_detector.focus_data = []
        focus_detector.frame_count = 0
        
        # 记录开始时间
        start_time = time.time()
        frame_count = 0
        written_count = 0
        focus_data_list = []
        processing_times = []  # 用于存储每帧处理时间
        inflight = deque()
        max_inflight = max(1, focus_detector.model_pool_size * 2)
        
        # 主处理循环
        while True:
            # 检查是否暂停
            while processing_status['is_paused']:
                time.sleep(0.5)
                continue
            
            ret, frame = cap.read()
            if not ret:
                break

            # 提交帧进行并行处理
            submit_time = time.time()
            try:
                future = focus_detector.process_frame_async(frame)
                inflight.append((frame_count, submit_time, future))
            except Exception as e:
                logger.error(f"提交帧处理任务失败: {e}")

            # 当在途任务超限时，取出最早结果并写入
            if len(inflight) > max_inflight:
                idx, st, fut = inflight.popleft()
                try:
                    vis_frame, frame_data = fut.result()
                except Exception as e:
                    logger.error(f"并行帧处理失败 (frame {idx}): {e}")
                    continue

                frame_processing_time = time.time() - st
                processing_times.append(frame_processing_time)

                # 写入视频
                out.write(vis_frame)

                # 计算平均专注度
                avg_focus = 0
                if frame_data['students']:
                    avg_focus = sum(student['focus_score'] for student in frame_data['students']) / len(frame_data['students'])

                # 添加到实时数据
                focus_data_list.append({
                    'frame': idx,
                    'focus': avg_focus
                })

                # 保存当前帧到临时文件，用于流式显示
                current_frame_path = os.path.join(temp_frames_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(current_frame_path, vis_frame)

                # 计算性能统计
                avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
                current_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

                # 更新处理状态 - 使用绝对路径
                elapsed_time = time.time() - start_time
                progress = (written_count / total_frames) * 100 if total_frames > 0 else 0
                processing_status['progress'] = progress
                processing_status['current_frame'] = idx
                processing_status['elapsed_time'] = elapsed_time
                processing_status['last_frame_path'] = os.path.abspath(current_frame_path)
                processing_status['focus_data'] = focus_data_list
                processing_status['average_focus'] = sum(item['focus'] for item in focus_data_list) / len(focus_data_list) if focus_data_list else 0
                processing_status['processing_fps'] = current_fps
                processing_status['avg_processing_time'] = avg_processing_time * 1000  # 转换为毫秒

                # 添加到CSV数据（以写入索引为准）
                if written_count % 5 == 0:
                    focus_detector._append_csv_data(frame_data, idx)
                written_count += 1

            frame_count += 1

            # 每10帧打印一次状态（基于已写入帧）
            if written_count % 10 == 0 and written_count > 0:
                avg_time_ms = sum(processing_times) / len(processing_times) * 1000 if processing_times else 0
                avg_fps = 1.0 / (sum(processing_times) / len(processing_times)) if processing_times else 0
                logger.info(f"Progress: {written_count}/{total_frames} ({processing_status['progress']:.1f}%) | FPS: {avg_fps:.2f} | 处理时间: {avg_time_ms:.1f}ms/帧")
        
        # 完成处理
        cap.release()
        out.release()

        # 刷新剩余在途任务
        while inflight:
            idx, st, fut = inflight.popleft()
            try:
                vis_frame, frame_data = fut.result()
            except Exception as e:
                logger.error(f"并行帧处理失败 (frame {idx}): {e}")
                continue
            current_frame_path = os.path.join(temp_frames_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(current_frame_path, vis_frame)
            out.write(vis_frame)
            processing_times.append(time.time() - st)
            written_count += 1
            processing_status['current_frame'] = idx
            processing_status['last_frame_path'] = os.path.abspath(current_frame_path)
            if written_count % 5 == 0:
                focus_detector._append_csv_data(frame_data, idx)
        
        # 保存CSV数据和生成曲线图
        if focus_detector.focus_data:
            focus_detector._save_csv(output_csv_path)
            
            # 从CSV生成注意力曲线图
            if os.path.exists(output_csv_path):
                try:
                    attention_df = pd.read_csv(output_csv_path)
                    if not attention_df.empty:
                        # 确保必要的列存在，否则创建一个基本的CSV
                        required_columns = ['frame', 'focus_score']
                        if not all(col in attention_df.columns for col in required_columns):
                            # 如果缺少必要的列，创建一个基本的CSV
                            basic_data = []
                            for idx, item in enumerate(focus_data_list):
                                basic_data.append({
                                    'frame': item['frame'],
                                    'focus_score': item['focus'],
                                    'student_id': 'all'
                                })
                            attention_df = pd.DataFrame(basic_data)
                            attention_df.to_csv(output_csv_path, index=False)
                            print(f"创建了基本的CSV数据: {output_csv_path}")
                        
                        # 生成曲线图
                        if 'student_id' in attention_df.columns and 'focus_score' in attention_df.columns and 'frame' in attention_df.columns:
                            agg_df = attention_df.groupby('frame')['focus_score'].mean().reset_index()
                            plot_attention_over_time(agg_df, output_plot_path, student_id='Average Attention')
                        else:
                            # 尝试使用基本列生成图表
                            plot_attention_over_time(attention_df, output_plot_path)
                except Exception as e:
                    print(f"处理CSV和生成图表时出错: {e}")
        
        # 标记处理完成
        processing_status['status'] = 'completed'

        # 计算统计信息并转换为results.html期望的格式
        raw_stats = focus_detector._calculate_statistics()
        
        # 为results.html页面准备统计信息
        processing_stats = {
            'total_frames_processed': frame_count,
            'average_focus_all': raw_stats.get('overall_avg_focus', 0.0),
            'processing_time_seconds': elapsed_time,
            'average_fps': 1.0 / (sum(processing_times) / len(processing_times)) if processing_times else 0,
            'average_processing_time_ms': sum(processing_times) / len(processing_times) * 1000 if processing_times else 0,
            'gpu_enabled': GPU_AVAILABLE
        }
        
        # 构建学生摘要信息
        students_summary = []
        student_avg_focus = raw_stats.get('student_avg_focus', {})
        student_emotions = raw_stats.get('student_dominant_emotions', {})
        
        for student_id, avg_focus in student_avg_focus.items():
            students_summary.append({
                'id': student_id,
                'average_focus': avg_focus,
                'dominant_emotion': student_emotions.get(student_id, 'unknown')
            })
        
        if students_summary:
            processing_stats['students_summary'] = students_summary
            
        # 将统计信息保存到processing_status，以便results路由可以使用
        processing_status['processing_stats'] = processing_stats
    
        # 关闭检测器的线程池以释放资源
        try:
            focus_detector.shutdown()
        except Exception:
            pass

    except Exception as e:
        logger.error(f"视频处理过程中出错: {e}")
        processing_status['status'] = 'error'
        processing_status['error'] = str(e)

@app.route('/results')
def results():
    # 显示处理结果页面
    global processing_status
    
    if processing_status['status'] != 'completed':
        return redirect(url_for('processing_view'))
    
    # 如果是流处理，优先使用已保存的统计信息
    if processing_status.get('is_stream') and processing_status.get('processing_stats'):
        stats = processing_status.get('processing_stats')
    else:
        # 计算统计信息
        raw_stats = focus_detector._calculate_statistics()
        
        # 转换统计信息格式以匹配模板期望的格式
        stats = {
            'total_frames_processed': raw_stats.get('total_frames', 0),
            'average_focus_all': raw_stats.get('overall_avg_focus', 0.0),
            'processing_time_seconds': processing_status.get('elapsed_time', 0.0),
            'gpu_enabled': GPU_AVAILABLE,
            'average_fps': processing_status.get('processing_fps', 0.0),
            'average_processing_time_ms': processing_status.get('avg_processing_time', 0.0)
        }
        
        # 构建学生摘要信息
        students_summary = []
        student_avg_focus = raw_stats.get('student_avg_focus', {})
        student_emotions = raw_stats.get('student_dominant_emotions', {})
        
        for student_id, avg_focus in student_avg_focus.items():
            students_summary.append({
                'id': student_id,
                'average_focus': avg_focus,
                'dominant_emotion': student_emotions.get(student_id, 'unknown')
            })
        
        if students_summary:
            stats['students_summary'] = students_summary
    
    # 从处理状态中获取输出文件路径
    output_video_filename = os.path.basename(processing_status['output_video_path']) if processing_status['output_video_path'] else None
    output_csv_filename = os.path.basename(processing_status['output_csv_path']) if processing_status['output_csv_path'] else None
    output_plot_filename = os.path.basename(processing_status['output_plot_path']) if processing_status['output_plot_path'] else None
    
    # 获取流模式相关信息
    is_stream = processing_status['is_stream']
    stream_type = processing_status.get('stream_type', None)
    stream_description = processing_status.get('stream_description', None)
    
    # 获取处理时间信息
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - processing_status['elapsed_time'])) if processing_status['elapsed_time'] > 0 else None
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) 
    
    # 获取性能信息
    gpu_enabled = processing_status.get('gpu_enabled', False)
    device_info = "GPU" if gpu_enabled else "CPU"
    
    return render_template('results.html',
                          original_video=os.path.basename(processing_status['video_path']) if processing_status['video_path'] else None,
                          processed_video=output_video_filename,
                          csv_data=output_csv_filename,
                          attention_plot=output_plot_filename,
                          stats=stats,
                          is_stream=is_stream,
                          stream_type=stream_type,
                          stream_description=stream_description,
                          start_time=start_time,
                          end_time=end_time,
                          device_info=device_info)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    try:
        # 尝试相对路径
        rel_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(rel_path):
            logger.debug(f"使用相对路径访问上传文件: {rel_path}")
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

        # 尝试绝对路径
        abs_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if not os.path.isfile(abs_path):
            # 尝试直接使用传入的路径
            if os.path.isfile(filename):
                directory = os.path.dirname(filename)
                basename = os.path.basename(filename)
                logger.debug(f"使用直接路径访问上传文件: {filename}")
                return send_from_directory(directory, basename)
            else:
                logger.warning(f"请求的上传文件不存在: {filename}")
                return f"文件不存在: {filename}", 404
                
        directory = os.path.dirname(abs_path)
        basename = os.path.basename(abs_path)
        
        logger.debug(f"使用绝对路径访问上传文件: {abs_path}")
        return send_from_directory(directory=directory, path=basename)
    except Exception as e:
        logger.error(f"访问上传文件时出错: {str(e)}")
        return f"访问文件时出错: {str(e)}", 500

@app.route('/output_data/<path:filename>')
def output_file(filename):
    """处理输出文件的访问，如处理后的视频、CSV数据和图表"""
    try:
        # 记录请求信息
        logger.debug(f"接收到文件请求: {filename}, 内容类型: {request.headers.get('Accept', '未指定')}")
        logger.debug(f"请求范围: {request.headers.get('Range', '未指定')}")
        
        # 获取文件扩展名并设置正确的MIME类型
        file_ext = os.path.splitext(filename)[1].lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.csv': 'text/csv',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        
        # 尝试找到文件的实际路径
        file_path = None
        tried_paths = []
        
        # 1. 检查相对路径
        rel_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        tried_paths.append(rel_path)
        if os.path.isfile(rel_path):
            file_path = rel_path
            logger.debug(f"文件存在于相对路径: {rel_path}")
        
        # 2. 检查绝对路径
        if not file_path:
            abs_path = os.path.abspath(os.path.join(app.config['OUTPUT_FOLDER'], filename))
            tried_paths.append(abs_path)
            if os.path.isfile(abs_path):
                file_path = abs_path
                logger.debug(f"文件存在于绝对路径: {abs_path}")
        
        # 3. 尝试直接使用传入的路径
        if not file_path and os.path.isfile(filename):
            file_path = filename
            tried_paths.append(filename)
            logger.debug(f"文件存在于指定路径: {filename}")
        
        # 如果找不到文件
        if not file_path:
            logger.warning(f"请求的文件不存在，尝试过以下路径:")
            for path in tried_paths:
                logger.warning(f" - {path}")
            return f"文件不存在: {filename}", 404
        
        # 如果是视频文件，使用专门的方法处理，支持范围请求(Range)
        if file_ext == '.mp4':
            logger.debug(f"使用特殊方法处理视频文件: {file_path}")
            return send_video_file(file_path)
        
        # 其他文件类型使用标准方法
        directory = os.path.dirname(file_path)
        basename = os.path.basename(file_path)
        
        response = send_from_directory(directory=directory, path=basename)
        
        # 确保设置正确的MIME类型
        if file_ext in mime_types:
            response.headers['Content-Type'] = mime_types[file_ext]
            logger.debug(f"设置MIME类型: {mime_types[file_ext]}")
            
        # 添加跨域访问头
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, Accept, Range'
        
        return response
    except Exception as e:
        logger.error(f"访问输出文件时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"访问文件时出错: {str(e)}", 500

def send_video_file(file_path):
    """特殊处理视频文件，支持范围请求"""
    try:
        video_size = os.path.getsize(file_path)
        logger.debug(f"视频文件大小: {video_size} bytes ({video_size/1024/1024:.2f} MB)")
        
        # 检查是否为范围请求
        range_header = request.headers.get('Range', None)
        
        if range_header:
            logger.debug(f"收到范围请求: {range_header}")
            byte_start, byte_end = 0, None
            
            if range_header.startswith('bytes='):
                range_values = range_header.replace('bytes=', '').split('-')
                if len(range_values) >= 1 and range_values[0]:
                    byte_start = int(range_values[0])
                if len(range_values) >= 2 and range_values[1]:
                    byte_end = int(range_values[1])
            
            if byte_end is None:
                byte_end = video_size - 1
                
            length = byte_end - byte_start + 1
            
            logger.debug(f"范围响应: 开始={byte_start}, 结束={byte_end}, 长度={length}")
            
            # 打开文件并读取指定范围
            with open(file_path, 'rb') as video_file:
                video_file.seek(byte_start)
                data = video_file.read(length)
                
            # 创建部分内容响应
            response = Response(
                data, 
                206,
                mimetype='video/mp4',
                content_type='video/mp4',
                direct_passthrough=True
            )
            
            # 设置范围响应头
            response.headers.add('Content-Range', f'bytes {byte_start}-{byte_end}/{video_size}')
            response.headers.add('Accept-Ranges', 'bytes')
            response.headers.add('Content-Length', str(length))
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Origin, Content-Type, Accept, Range')
            
            logger.debug("视频范围请求响应头:")
            for header, value in response.headers:
                logger.debug(f" - {header}: {value}")
                
            return response
        else:
            # 非范围请求，返回整个文件
            logger.debug("非范围请求，返回整个视频文件")
            directory = os.path.dirname(file_path)
            basename = os.path.basename(file_path)
            
            response = send_from_directory(
                directory=directory, 
                path=basename,
                mimetype='video/mp4'
            )
            
            response.headers.add('Accept-Ranges', 'bytes')
            response.headers.add('Content-Length', str(video_size))
            response.headers.add('Access-Control-Allow-Origin', '*')
            
            logger.debug("视频完整响应头:")
            for header, value in response.headers:
                logger.debug(f" - {header}: {value}")
                
            return response
            
    except Exception as e:
        logger.error(f"处理视频文件时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return f"处理视频文件时出错: {str(e)}", 500

@app.route('/temp_data/<path:filepath>')
def temp_file(filepath):
    """处理临时文件的访问，支持任意嵌套的目录结构"""
    try:
        # 记录路径用于调试
        logger.debug(f"请求的临时文件路径: {filepath}")
        
        # 尝试相对路径
        if '/' in filepath:
            directory, filename = filepath.rsplit('/', 1)
            rel_directory_path = os.path.join(app.config['TEMP_FOLDER'], directory)
            rel_file_path = os.path.join(rel_directory_path, filename)
            if os.path.isfile(rel_file_path):
                return send_from_directory(rel_directory_path, filename)
        else:
            filename = filepath
            rel_file_path = os.path.join(app.config['TEMP_FOLDER'], filename)
            if os.path.isfile(rel_file_path):
                return send_from_directory(app.config['TEMP_FOLDER'], filename)
        
        # 尝试绝对路径
        abs_path = None
        if os.path.isabs(filepath) and os.path.isfile(filepath):
            abs_path = filepath
        else:
            abs_path = os.path.abspath(os.path.join(app.config['TEMP_FOLDER'], filepath))
            
        if not os.path.isfile(abs_path):
            logger.warning(f"请求的临时文件不存在: {abs_path}")
            return f"临时文件不存在: {filepath}", 404
            
        directory = os.path.dirname(abs_path)
        basename = os.path.basename(abs_path)
        
        logger.debug(f"尝试从 {directory} 读取临时文件 {basename}")
        return send_from_directory(directory, basename)
    except Exception as e:
        logger.error(f"访问临时文件时出错: {str(e)}")
        return f"访问临时文件时出错: {str(e)}", 500

# 提供模板目录中的静态资源访问（如占位图）
@app.route('/templates/<path:filename>')
def templates_file(filename):
    try:
        template_dir = os.path.abspath('templates')
        file_path = os.path.join(template_dir, filename)
        if not os.path.isfile(file_path):
            logger.warning(f"请求的模板文件不存在: {file_path}")
            return f"模板文件不存在: {filename}", 404
        return send_from_directory(template_dir, filename)
    except Exception as e:
        logger.error(f"访问模板文件时出错: {str(e)}")
        return f"访问模板文件时出错: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=False)
    os.rmdir(app.config['TEMP_FOLDER'])
    os.rmdir(app.config['UPLOAD_FOLDER'])