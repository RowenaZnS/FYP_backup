#!/usr/bin/env python3
"""
调试工具：附加到运行中的训练进程并打印堆栈跟踪
使用方法: python debug_trace.py
"""

import sys
import os
import signal
import subprocess

def find_train_processes():
    """找到所有训练进程"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        processes = []
        for line in result.stdout.split('\n'):
            if 'train.py' in line and 'local-rank' in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    cmd = ' '.join(parts[10:])
                    processes.append((pid, cmd))
        return processes
    except Exception as e:
        print(f"Error finding processes: {e}")
        return []

def print_stack_trace(pid):
    """使用 gdb 打印堆栈跟踪"""
    print(f"\n正在附加到进程 {pid}...")
    print("=" * 60)
    
    gdb_commands = """
import sys
sys.path.insert(0, '/workspace/SwinFace/swinface_project')
import traceback
import threading

# 打印所有线程的堆栈
for thread_id, frame in sys._current_frames().items():
    print(f"\\nThread {thread_id}:")
    traceback.print_stack(frame)
"""
    
    try:
        # 使用 gdb 执行 Python 代码
        cmd = f'gdb -p {pid} -batch -ex "python {gdb_commands}" -ex "detach" -ex "quit"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except subprocess.TimeoutExpired:
        print("超时：进程可能正在运行，无法附加")
    except Exception as e:
        print(f"错误: {e}")
        print("\n尝试使用 py-spy:")
        print(f"  py-spy dump --pid {pid}")

if __name__ == "__main__":
    print("查找训练进程...")
    processes = find_train_processes()
    
    if not processes:
        print("未找到训练进程")
        sys.exit(1)
    
    print(f"找到 {len(processes)} 个进程:")
    for i, (pid, cmd) in enumerate(processes, 1):
        print(f"  {i}. PID: {pid}")
        print(f"     命令: {cmd[:80]}...")
    
    # 默认查看第一个进程（通常是 rank 0）
    if processes:
        pid = processes[0][0]
        print_stack_trace(pid)


