#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
'''
@File    :   stl_rename.py
@Time    :   2025/07/24 11:32:38
@Author  :   StarCheng
@Version :   1.0
@Site    :   https://star-cheng.github.io/Blog/
'''
import os

stl_dir = r"./description/tiangong_description/meshes_back"
stl_listdir = os.listdir(stl_dir)

for stl_name in stl_listdir:
    rename = stl_name.split(".")[0] + ".stl"
    stl_path = os.path.join(stl_dir, stl_name)
    rename_path = os.path.join(stl_dir, rename)
    os.rename(stl_path, rename_path)
    print(stl_name, "rename to", rename)