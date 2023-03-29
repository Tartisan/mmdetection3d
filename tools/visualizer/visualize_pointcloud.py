import open3d
import pypcd
import numpy as np
import argparse
import os

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('data', type=str, default='data', 
                        help='specify the point cloud data file or directory')
    parser.add_argument('--dims', type=int, default=5, 
                        help='x, y, z, intensity, [timestamp], [elongation]')
    args = parser.parse_args()
    return args


def main():
    args = parse_config()

    data_file = args.data
    if os.path.splitext(data_file)[-1] == ".bin":
        bin = np.fromfile(data_file, dtype=np.float32)
        data = bin.reshape((bin.size // args.dims, args.dims))
    elif os.path.splitext(data_file)[-1] == ".pcd":
        pcd = pypcd.PointCloud.from_path(data_file)
        data = np.stack([pcd.pc_data['x'], pcd.pc_data['y'], pcd.pc_data['z']])
        for field in ['intensity', 'timestamp', 'elongation']:
            if field in pcd.fields:
                tmp = pcd.pc_data[field].reshape(1, len(pcd.pc_data[field]))
                data = np.concatenate((data, tmp), axis=0)
        data = data.transpose(1, 0)
    print("pointcloud shape: ", data.shape)
    print(data)
    points = data[:, 0:3]
    intensity = data[:, 3]
    # intensity to rgb colors
    print("max_intensity: ", max(intensity))
    scale = 1 if max(intensity) > 1 else 255
    intensity = (intensity * scale).astype(int)
    colors = pypcd.decode_rgb_from_pcl(intensity)

    vis = open3d.visualization.Visualizer()
    vis.create_window()
    # point size
    vis.get_render_option().point_size = 2.0
    # background color
    vis.get_render_option().background_color = np.zeros(3)
    # origin
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)
    # points
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points)
    pc.colors = open3d.utility.Vector3dVector(colors)
    vis.add_geometry(pc)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()