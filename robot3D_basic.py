import numpy as np
from vedo import *
import time

def RotationMatrix(theta, axis_name):
    """ calculate single rotation of $theta$ matrix around x,y or z
        code from: https://programming-surgeon.com/en/euler-angle-python-en/
    input
        theta = rotation angle(degrees)
        axis_name = 'x', 'y' or 'z'
    output
        3x3 rotation matrix
    """

    c = np.cos(theta * np.pi / 180)
    s = np.sin(theta * np.pi / 180)

    if axis_name == 'x':
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, c, -s],
                                    [0, s, c]])
    if axis_name == 'y':
        rotation_matrix = np.array([[c, 0, s],
                                    [0, 1, 0],
                                    [-s, 0, c]])
    elif axis_name == 'z':
        rotation_matrix = np.array([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]])
    return rotation_matrix


def createCoordinateFrameMesh():
    """Returns the mesh representing a coordinate frame
    Args:
      No input args
    Returns:
      F: vedo.mesh object (arrows for axis)
    """
    _shaft_radius = 0.05
    _head_radius = 0.10
    _alpha = 1

    # x-axis as an arrow
    x_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(1, 0, 0),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='red',
                        alpha=_alpha)

    # y-axis as an arrow
    y_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(0, 1, 0),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='green',
                        alpha=_alpha)

    # z-axis as an arrow
    z_axisArrow = Arrow(start_pt=(0, 0, 0),
                        end_pt=(0, 0, 1),
                        s=None,
                        shaft_radius=_shaft_radius,
                        head_radius=_head_radius,
                        head_length=None,
                        res=12,
                        c='blue',
                        alpha=_alpha)

    originDot = Sphere(pos=[0, 0, 0],
                       c="black",
                       r=0.10)

    # Combine the axes together to form a frame as a single mesh object
    F = x_axisArrow + y_axisArrow + z_axisArrow + originDot

    return F


def getLocalFrameMatrix(R_ij, t_ij):
    """Returns the matrix representing the local frame
    Args:
      R_ij: rotation of Frame j w.r.t. Frame i
      t_ij: translation of Frame j w.r.t. Frame i
    Returns:
      T_ij: Matrix of Frame j w.r.t. Frame i.
    """
    # Rigid-body transformation [ R t ]
    T_ij = np.block([[R_ij, t_ij],
                     [np.zeros((1, 3)), 1]])

    return T_ij


def main():
    # Set the limits of the graph x, y, and z ranges
    axes = Axes(xrange=(0, 20), yrange=(-2, 10), zrange=(0, 6))

    # Lengths of arm parts
    L1 = 5  # Length of link 1
    L2 = 8  # Length of link 2

    show(axes, interactive=False)

    # joint angles
    phi1 = 0
    phi2 = 0
    phi3 = 0

    # Create the coordinate frame meshes for each frame
    Frame1Arrows = createCoordinateFrameMesh()
    Frame2Arrows = createCoordinateFrameMesh()
    Frame3Arrows = createCoordinateFrameMesh()

    # Create cylinder meshes for links
    link1_mesh = Cylinder(r=0.4, height=L1, pos=(L1 / 2, 0, 0), c="yellow", alpha=.8, axis=(1, 0, 0))
    link2_mesh = Cylinder(r=0.4, height=L2, pos=(L2 / 2, 0, 0), c="red", alpha=.8, axis=(1, 0, 0))

    max_frames = 30

    for frames in range(max_frames):
        R_01 = RotationMatrix(phi1, axis_name='z')  # Rotation matrix
        p1 = np.array([[3], [2], [0.0]])
        t_01 = p1

        T_01 = getLocalFrameMatrix(R_01, t_01)

        R_12 = RotationMatrix(phi2, axis_name='z')
        p2 = np.array([[L1], [0.0], [0.0]])
        t_12 = p2

        T_12 = getLocalFrameMatrix(R_12, t_12)

        T_02 = T_01 @ T_12

        R_23 = RotationMatrix(phi3, axis_name='z')
        p3 = np.array([[L2], [0.0], [0.0]])
        t_23 = p3

        T_23 = getLocalFrameMatrix(R_23, t_23)

        T_03 = T_01 @ T_12 @ T_23

        Frame1 = Frame1Arrows.clone().apply_transform(T_01)
        Frame2 = Frame2Arrows.clone().apply_transform(T_02)
        Frame3 = Frame3Arrows.clone().apply_transform(T_03)

        link1 = link1_mesh.clone().apply_transform(T_01)
        link2 = link2_mesh.clone().apply_transform(T_02)

        # Show the current pose
        show([Frame1, Frame2, Frame3, link1, link2], at=0, axes=0, interactive=False, viewup="z")

        # Update joint angles for next iteration
        phi1 += 1  # Adjust the increment as needed
        phi2 += 1  # Adjust the increment as needed
        phi3 += 1  # Adjust the increment as needed

        time.sleep(0.1)  


if __name__ == '__main__':
    main()
