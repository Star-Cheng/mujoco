#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
'''
@File    :   numpy_trans.py
@Time    :   2025/07/24 11:32:30
@Author  :   StarCheng
@Version :   1.0
@Site    :   https://star-cheng.github.io/Blog/
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
'''
@File    :   numpy_trans.py
@Time    :   2025/07/24 09:25:13
@Author  :   StarCheng
@Version :   1.0
@Site    :   https://star-cheng.github.io/Blog/
'''
from stl import mesh
import numpy as np
import os

stl_dir = r"./description/tiangong_description/meshes_back"
save_dir = r"./description/tiangong_description/meshes"
stl_listdir = os.listdir(stl_dir)
for stl_name in stl_listdir:
    input_file = os.path.join(stl_dir, stl_name)
    output_file = os.path.join(save_dir, stl_name)

    # 读取STL（自动处理ASCII/二进制）
    your_mesh = mesh.Mesh.from_file(input_file, mode=2)

    # 保存为二进制格式
    your_mesh.save(output_file)

    print(f"Converted {input_file} to binary: {output_file}")