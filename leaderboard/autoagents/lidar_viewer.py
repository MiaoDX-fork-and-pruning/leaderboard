import open3d as o3d

# 读取PCD文件
pcd = o3d.io.read_point_cloud("received_lidar_1744093577.pcd")

# 可视化点云
o3d.visualization.draw_geometries([pcd])
