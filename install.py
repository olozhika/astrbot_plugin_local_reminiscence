import subprocess
import sys
import importlib.util

def check_package(package_name):
    """检查包是否已安装"""
    return importlib.util.find_spec(package_name) is not None

def main():
    # 自动获取当前运行的Python解释器路径
    python_path = sys.executable
    print(f"使用Python解释器: {python_path}")
    
    # 检查 torch
    if not check_package('torch'):
        print("正在安装 PyTorch CPU 版本...")
        try:
            subprocess.check_call([
                python_path, '-m', 'pip', 'install',
                'torch', 'torchvision',
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ])
            print("PyTorch CPU 安装完成！")
        except subprocess.CalledProcessError as e:
            print(f"PyTorch 安装失败: {e}")
            sys.exit(1)
    else:
        print("PyTorch 已安装，跳过...")
    
    # 安装其他依赖
    print("正在安装其他依赖...")
    try:
        subprocess.check_call([
            python_path, '-m', 'pip', 'install',
            '-r', 'requirements.txt'
        ])
        print("所有依赖安装完成！")
    except subprocess.CalledProcessError as e:
        print(f"依赖安装失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
