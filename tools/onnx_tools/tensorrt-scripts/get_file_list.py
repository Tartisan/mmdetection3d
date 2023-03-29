# -*- coding: UTF-8 -*-
#!/usr/bin/env python3
import os
import os.path as osp


def main():
    for _, _, file_list in os.walk(osp.abspath('work_dirs/quat')):
        file_list.sort()
        with open('mb-bp_val.txt', 'w+') as f:
            for idx in file_list:
                f.write(str(idx) + '\n')


if __name__ == '__main__':
    main()