# -------- Library importings -------- #
# Importing system libraries
import glob
import os

# and math, signal and plot libraries for creating user-defined functions
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# -------- Error handling classes -------- #
# - ShapeError
# - DimensionError
# - TransformTypeError

class ShapeError(Exception):
    """If image shapes are not equal raise"""
    pass


class DimensionError(Exception):
    """If number of dimension is not equal to 3 raise"""
    pass


class TransformTypeError(Exception):
    """If number of dimension is not equal to 3 raise"""
    pass


# -------- Joint Histogram Function -------- #
# - Joint Histogram

def joint_histogram(arr1, arr2, bins, plot=False, save=None):
    # Flattening arrays
    r_arr1 = np.ravel(arr1)
    r_arr2 = np.ravel(arr2)
    # Combine two 1D arrays into one in order to work with multi-dimensional
    sample = np.array([r_arr1, r_arr2]).T
    num, dimension = sample.shape

    n_bin = np.zeros(dimension, dtype=np.int16)
    edges = [None] * dimension
    diff_edges = [None] * dimension
    bins_list = [bins] * dimension

    # Get edges for every dimension and get differentiate it
    for dim_i in range(dimension):
        s_min, s_max = sample[:, dim_i].min(), sample[:, dim_i].max()

        edges[dim_i] = np.linspace(s_min, s_max, bins_list[0] + 1)
        n_bin[dim_i] = len(edges[dim_i]) + 1
        diff_edges[dim_i] = np.diff(edges[dim_i])

    n_count = [np.searchsorted(edges[dim_i], sample[:, dim_i], side='right') for dim_i in range(dimension)]

    # Check bins on edge
    for dim_i in range(dimension):
        on_edge = (sample[:, dim_i] == edges[dim_i][-1])
        n_count[dim_i][on_edge] -= 1

    # Linear index into your array
    XY = np.ravel_multi_index(n_count, n_bin)
    # Count bins, total number of bin is their products
    hist = np.bincount(XY, minlength=np.product(n_bin))
    # Assign as float32 and reshape with n_bin
    hist = hist.reshape(n_bin).astype(np.float32)
    # Remove 0 and -1 indices from all dimensions
    outliers = (slice(1, -1),) * dimension
    hist = hist[outliers]

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
        ax[0].imshow(arr1, cmap="gray")
        ax[0].set_title("Image I")
        ax[1].imshow(arr2, cmap="gray")
        ax[1].set_title("Image J")
        ax[2].imshow(hist, cmap="gray", origin="lower")
        ax[2].set_title("Joint Histogram of I and J")
        if save is not None:
            plt.savefig(save)
        plt.show()
    return hist, edges


# -------- Similarity Functions -------- #
# - SSD
# - Correlation
# - Mutual Information

def SSD(arr1, arr2):
    return np.sum((arr1[:, :] - arr2[:, :]) ** 2)


def correlation(arr1, arr2):
    mean_x, mean_y = arr1.mean(), arr2.mean()
    # Calculating difference of a point from mean
    x_diff_mean = arr1 - mean_x
    y_diff_mean = arr2 - mean_y

    coeff_num = np.sum(x_diff_mean * y_diff_mean)
    coeff_den = np.sqrt(np.sum(x_diff_mean * x_diff_mean) * np.sum(y_diff_mean * y_diff_mean))
    coeff = coeff_num / coeff_den
    return coeff


def MI(arr1, arr2):
    # The images passed will first go to joint histogram function to get the joint histogram of two images
    hist, _ = joint_histogram(arr1, arr2, bins=50)
    # Converting the his2d zero element to probability values
    pxy = hist / float(np.sum(hist))
    # Generating marginal sof x and y over each other
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi


# -------- Transformation Kernels and Functions -------- #
# - Kernel function
# - Transformation ("Rigid" or "Affine")
# - Applying transformation
# - Translation with interpolation

def _create_transformation_kernel(theta, omega, phi, s=None):
    rot_x_theta = np.array([[1, 0, 0, 0],
                            [0, np.cos(theta), -np.sin(theta), 0],
                            [0, np.sin(theta), np.cos(theta), 0],
                            [0, 0, 0, 1]])

    rot_y_omega = np.array([[np.cos(omega), 0, np.sin(omega), 0],
                            [0, 1, 0, 0],
                            [-np.sin(omega), 0, np.cos(omega), 0],
                            [0, 0, 0, 1]])

    rot_z_phi = np.array([[np.cos(phi), -np.sin(phi), 0, 0],
                          [np.sin(phi), np.cos(phi), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    if s is not None:
        scaled_matrix = np.array([[s, 0, 0, 0],
                                  [0, s, 0, 0],
                                  [0, 0, s, 0],
                                  [0, 0, 0, 1]])

        return rot_x_theta, rot_y_omega, rot_z_phi, scaled_matrix
    return rot_x_theta, rot_y_omega, rot_z_phi


def transform(type, theta, omega, phi, p, q, r, s=None):
    # Checking transformation type
    try:
        if not (type == "rigid" or type == "affine"):
            raise TransformTypeError
    except TransformTypeError as err:
        print("Entered wrong transformation parameter. Only `rigid` and `affine` supported")

    try:
        if type == "affine" and s is None:
            raise TransformTypeError
    except TransformTypeError as err:
        print("s is `None`, Affine transformation needs `s` parameter")

    if type == "rigid":
        rot_x_theta, rot_y_omega, rot_z_phi = _create_transformation_kernel(theta, omega, phi)
    else:
        rot_x_theta, rot_y_omega, rot_z_phi, scaled_matrix = _create_transformation_kernel(theta, omega, phi, s=s)

    # Angle transformation
    transf_angle = np.matmul(rot_x_theta, rot_y_omega, rot_z_phi)
    if type == "affine":
        # Scale transformation
        scaled_transf = np.matmul(transf_angle, scaled_matrix)
        transf_matrix = np.copy(scaled_transf)
    else:
        transf_matrix = np.copy(transf_angle)

    transf_matrix[0, 3] = p
    transf_matrix[1, 3] = q
    transf_matrix[2, 3] = r

    return (rot_x_theta, rot_y_omega, rot_z_phi), (transf_angle, transf_matrix)


def apply_transformation(matrix, type, theta, omega, phi, p, q, r, s=None):
    matrix_transpose = matrix.T  # Transposing the matrix in a
    ones = np.ones((1, matrix_transpose.shape[1]))
    # adding row of ones to make it 4 dim
    trans_matrix = np.append(matrix_transpose, ones, axis=0)
    rot, trans = transform(type=type,
                           theta=theta, omega=omega, phi=phi,
                           p=p, q=q, r=r, s=s)

    # The transformation matrix will be used for rigid transformation on a
    trans_points = np.empty((0, trans_matrix.shape[0]), float)

    for i in range(matrix_transpose.shape[1]):
        points = np.matmul(trans[1], trans_matrix[:, i])
        trans_points = np.append(trans_points, np.array([points]), axis=0)

    # the trans_points will have 4 columns with last column as 1, thus removing the 4th column and draw the grid
    x = trans_points[:, 0]
    y = trans_points[:, 1]
    z = trans_points[:, 2]

    return x, y, z


def translation(arr, p, q):
    x = np.linspace(0, arr.shape[0], arr.shape[0])
    y = np.linspace(0, arr.shape[1], arr.shape[1])

    f = interp2d(x + p, y + q, arr, kind='cubic', fill_value=0)
    z = f(x, y)
    return z


def rotate_coords(x, y, theta, ox, oy):
    """Rotate arrays of coordinates x and y by theta radians about the
    point (ox, oy).

    """
    s, c = np.sin(theta), np.cos(theta)
    x, y = np.asarray(x) - ox, np.asarray(y) - oy
    return x * c - y * s + ox, x * s + y * c + oy


def rotation(arr, theta, ox=0, oy=0, fill=0):
    """Rotate the image src by theta radians about (ox, oy).
    Pixels in the result that don't correspond to pixels in src are
    replaced by the value fill.

    """
    # Images have origin at the top left, so negate the angle.
    theta = theta * np.pi / 180
    theta = -theta
    # Dimensions of source image. Note that scipy.misc.imread loads
    # images in row-major order, so src.shape gives (height, width).
    sw, sh = arr.shape

    # Rotated positions of the corners of the source image.
    cx, cy = rotate_coords([0, sw, sw, 0], [0, 0, sh, sh], theta, ox, oy)

    # Determine dimensions of destination image.
    dw, dh = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))

    # Coordinates of pixels in destination image.
    dx, dy = np.meshgrid(np.arange(dw), np.arange(dh))

    # Corresponding coordinates in source image. Since we are
    # transforming dest-to-src here, the rotation is negated.
    sx, sy = rotate_coords(dx, dy, -theta, ox, oy)

    # Select nearest neighbour.
    sx, sy = sx.round().astype(int), sy.round().astype(int)

    # Mask for valid coordinates.
    mask = (0 <= sx) & (sx < sw) & (0 <= sy) & (sy < sh)

    # Create destination image.
    dest = np.empty(shape=(dh, dw), dtype=arr.dtype)
    # Copy valid coordinates from source image.
    dest[dy[mask], dx[mask]] = arr[sy[mask], sx[mask]]

    # Fill invalid coordinates.
    dest[dy[~mask], dx[~mask]] = fill
    dest = dest[:256, :256]
    return dest


# -------- Optimization Functions -------- #
# - Basic min loss find func

def find_min_SSD(func, ref, target, vectors, tot_step=20):
    try:
        if func == 'translation':
            init_v = [0, 0]
        elif func == 'rotation':
            pass
        else:
            raise TypeError

    except TypeError as err:
        print('Only `translation` or `rotation` keywords supported for this function')
        print('`translation` keyword going to be used in this function for now')
        return

    ssd_hist = [np.float64(SSD(ref, target))]
    if func == 'translation':
        for step in range(tot_step):
            ssd_local = []
            for v in vectors:
                target_test = translation(target, init_v[0] + v[0], init_v[1] + v[1])
                ssd_local.append(SSD(ref, target_test))

            init_v = np.add(np.array(vectors[np.argmin(ssd_local)]), init_v)
            ssd_hist.append(ssd_local[np.argmin(ssd_local)])
        new_target = translation(target, init_v[0], init_v[1])
        return ssd_hist, new_target, init_v

    if func == 'rotation':
        ssd_local = []
        for d in vectors:
            target_test = rotation(target, theta=d, ox=0, oy=0, fill=0)
            ssd_local.append(SSD(ref, target_test))

        min_d = np.array(vectors[np.argmin(ssd_local)])
        new_target = rotation(target, theta=min_d)
        return ssd_local, new_target, min_d


def gradient_descent(source, target):
    init_tx = 5
    init_ty = 0
    init_degree = 10
    step_size = 0.001
    step_tolerance = 0.000001
    max_iterations = 2000

    source = source * (1/255)
    target = target * (1/255)
    source = translation(source, p=0, q=0)
    ssd_init = SSD(source, target)

    for epoch in range(max_iterations):
        trans_x_target = translation(target, p=init_tx, q=0)
        ssd_trans_x = SSD(source, trans_x_target)
        tx_step = step_size * np.diff([ssd_init, ssd_trans_x])
        init_tx -= tx_step

        trans_y_target = translation(trans_x_target, p=0, q=init_ty)
        ssd_trans_y = SSD(source, trans_y_target)
        ty_step = step_size * np.diff([ssd_init, ssd_trans_y])
        init_ty -= ty_step

        rot_target = rotation(trans_y_target, init_degree)
        ssd_rot = SSD(source, rot_target)
        rot_step = step_size * np.diff([ssd_init, ssd_rot])
        init_degree -= rot_step
        init_degree %= 360

        if epoch % 50 == 0:
            print(f"Epoch: {epoch}")
            print(f"SSD loss rotation: {ssd_rot}, rotation_degree = {init_degree}")
            print(f"SSD loss translation on x: {ssd_trans_x}, orientation x = {init_tx}")
            print(f"SSD loss translation on y: {ssd_trans_y}, orientation y = {init_ty}\n")

        if epoch % 400 == 0:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(source * 255, cmap="gray")
            ax[1].imshow(rot_target * 255, cmap="gray")
            plt.title(f"Epoch: {epoch}")
            plt.show()

        if abs(rot_step) <= step_tolerance and abs(tx_step) <= step_tolerance and abs(ty_step) <= step_tolerance:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(source * 255, cmap="gray")
            ax[1].imshow(rot_target * 255, cmap="gray")
            plt.title(f"Epoch: {epoch}")
            plt.show()
            break


def create_pos_vector():
    pos_left_down = (-0.5, -0.5)
    pos_left_up = (-0.5, 0.5)
    pos_right_down = (0.5, -0.5)
    pos_right_up = (0.5, 0.5)
    return pos_left_down, pos_left_up, pos_right_down, pos_right_up


def create_rot_degree():
    return np.arange(-180, +180, 1)


# -------- File Manipulation and Read Functions -------- #
# - Reading images from paths
# - Checking 3rd dimension
# - Converting grayscale from RGB

def read_images(I_path, J_path, MRI_path):
    I = []
    J = []
    MRI = []

    for I_p, J_p in zip(I_path, J_path):
        imI = mpimg.imread(I_p)
        if np.ndim(imI) == 3:
            imI = rgb_to_gray(imI)
        imJ = mpimg.imread(J_p)
        if np.ndim(imJ) == 3:
            imJ = rgb_to_gray(imJ)
        if check_dimension(imI, imJ):
            I.append(imI)
            J.append(imJ)

    for MRI_p in MRI_path:
        imMRI = mpimg.imread(MRI_p)
        if np.ndim(imMRI) == 3:
            imMRI = rgb_to_gray(imMRI)
        MRI.append(imMRI)

    return I, J, MRI


def check_dimension(arr1, arr2):
    # Checking image shapes
    try:
        if arr1.shape[:2] != arr2.shape[:2]:
            raise ShapeError
        else:
            return True
    except ShapeError as err:
        print("Image shapes are not equal")
        print(f"Image 1: {arr1.shape}\nImage 2: {arr2.shape}")


def rgb_to_gray(arr):
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


# -------- Mesh and Plot Functions -------- #
# - Creating mesh elements
# - Plotting mesh scatter
# - Plotting mesh and rigid scatter
# - Plotting mesh and affine scatter
# - Plotting translation
# - Plotting rotation (alias of translation)

def create_mesh_elements(x_range, y_range, z_range):
    x = [a for a in range(x_range)]
    y = [b for b in range(y_range)]
    z = [c for c in range(z_range)]
    xy = []

    for k in z:
        for i in x:
            for j in y:
                point = [i, j, k]
                xy = np.append(xy, point)

    B = np.reshape(xy, (-1, 3))
    x = B[:, 0]
    y = B[:, 1]
    z = B[:, 2]

    return (x, y, z), B


def plot_mesh_scatter(x, y, z, save=None):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection='3d')
    ax.set_zlim(0, 20)
    ax.scatter(x, y, z, c='black', marker='o')
    ax.view_init(30, None)
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_mesh_rigid_scatter(x, y, z, x1, y1, z1, save=None):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection='3d')
    ax.set_zlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xlim(20, 0)
    ax.scatter(x, y, z, c='black', marker='o')
    ax.scatter(x1, y1, z1, c='blue', marker='o')
    ax.view_init(30, None)
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_mesh_affine_scatter(x, y, z, x1, y1, z1, save=None):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection='3d')
    ax.set_zlim(0, 20)
    ax.set_ylim(-10, 30)
    ax.set_xlim(30, -10)
    ax.scatter(x, y, z, c='black', marker='o')
    ax.scatter(x1, y1, z1, c='blue', marker='o')
    ax.view_init(30, None)
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_translation(arr1, arr2, save=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(arr1, cmap="gray")
    ax[1].imshow(arr2, cmap="gray")
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_hist_compare(arr1, arr2, hist, save=None):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(arr1, cmap="gray")
    ax[1].imshow(arr2, cmap="gray")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].plot(hist)
    if len(hist) == 360:
        ax[2].set_xticks(np.arange(0, 360, 45))
        ax[2].set_xlabel("Degree")
    else:
        ax[2].set_xticks(np.arange(0, len(hist), 20))
        ax[2].set_xlabel("Step")

    ax[2].set_ylabel("SSD Loss")

    if save is not None:
        plt.savefig(save)
    plt.show()


plot_degree_compare = plot_hist_compare

# Creating alias for rotation plotting
plot_rotation = plot_translation

if __name__ == "__main__":
    # ----------------------------  Initialize ---------------------------- #
    # Main paths of files and directories
    imI_path = sorted(glob.glob("data/I*.*"))
    imJ_path = sorted(glob.glob("data/J*.*"))
    imMRI_path = sorted(glob.glob("data/Brain*.*"))
    output_path = "output"
    print("Reading images from disk and saving arrays.\n\n")
    I_images, J_images, MRI_images = read_images(imI_path, imJ_path, imMRI_path)
    # ---------------------------- Part 1 - Joint histogram ---------------------------- #
    for i in range(len(I_images)):
        save_path_hist = os.path.join("output/joint_hist", f"I-J-{i + 1}.png")
        print(f"Pair: I-J-{i + 1}")
        # Calculating joint histogram of each I-J pair
        hist, _ = joint_histogram(I_images[i], J_images[i], bins=50, plot=True, save=save_path_hist)
        # Sum of Joint histogram coordinates
        sum_hist = sum(sum(hist))
        # The multiplication of input array shape
        mult_array = np.prod(I_images[i].shape)
        # The question is to confirm that the above two variables generated are equal
        print(f"The sum of 2d histogram coordinates values\n"
              f"and the multiplication of shape of input image\n"
              f"sum_hist == mult_array: {sum_hist == mult_array}")
        # ---------------------------- Part 2 - Similarity Criteria ---------------------------- #
        # Comparing the results of above functions
        ssd = SSD(I_images[i], J_images[i])
        corr_coeff = correlation(I_images[i], J_images[i])
        mi = MI(I_images[i], J_images[i])

        print(f"\tSSD between two images = {ssd:0.2f}")
        print(f"\tCorrelation between images = {corr_coeff:0.5f}")
        print(f"\tMutual information images = {mi:0.5f}\n\n")

    # ---------------------------- Part 3 - Spatial Transforms ---------------------------- #
    # -------- Section 1
    # Creating required mesh elements to using for transformations
    (x, y, z), B = create_mesh_elements(x_range=21, y_range=21, z_range=6)

    save_path_mesh = "output/transformations/mesh.png"
    plot_mesh_scatter(x, y, z, save_path_mesh)

    # -------- Section 2 - Rigid Transformation
    save_path_mesh_rigid = "output/transformations/mesh_rigid.png"
    x_rigid, y_rigid, z_rigid = apply_transformation(matrix=B, type='rigid',
                                                     theta=45, omega=45, phi=45,
                                                     p=20, q=20, r=0)

    plot_mesh_rigid_scatter(x, y, z, x_rigid, y_rigid, z_rigid, save_path_mesh_rigid)

    # -------- Section 3 - Affine Transformation
    save_path_mesh_affine = "output/transformations/mesh_affine.png"
    x_affine, y_affine, z_affine = apply_transformation(matrix=B, type='affine',
                                                        theta=45, omega=45, phi=90,
                                                        p=-10, q=-10, r=0, s=0.5)

    plot_mesh_affine_scatter(x, y, z, x_affine, y_affine, z_affine, save_path_mesh_affine)

    # ---------------------------- Part 4 - Simple 2D Registration  ---------------------------- #
    # Selecting image number randomly
    im_no_trans = 1
    im_no_rot = 2

    # -------- Section 1 - Translation
    trans_new = translation(MRI_images[im_no_trans], p=20, q=20)
    save_path_translation = f"output/registration/translation.png"
    plot_translation(MRI_images[0], trans_new, save_path_translation)

    # -------- Section 2 - Rotation
    rot_new = rotation(MRI_images[im_no_rot], theta=10, ox=0, oy=0, fill=0)
    save_path_rotation = f"output/registration/rotation.png"
    plot_rotation(MRI_images[0], rot_new, save_path_rotation)

    # -------- Section 3 - 2D registrations
    reference = MRI_images[0]
    targets = MRI_images[1:]
    total_step = 60

    # - Translation and Rotation with custom minimizing function
    vec = create_pos_vector()
    deg = create_rot_degree()

    print(f"Minimizing SSD with custom function")
    print(f"Total step: {total_step}")

    for image_index, tar in enumerate(targets):
        print(f"\tBrainMRI_1 - BrainMRI_{2 + image_index}")
        print(f"\tTranslation:")
        ssd_hist_trans, relocated_target, orientation_vec = find_min_SSD('translation', reference, tar,
                                                                         vectors=vec, tot_step=total_step)
        save_path_compare_trans = f"output/registration/translation_MRI1_MRI{2 + image_index}.png"
        plot_hist_compare(reference, relocated_target, ssd_hist_trans[1:], save_path_compare_trans)
        print(f"\t\tOrientation vector: {orientation_vec}")
        print(f"\t\tMin SSD: {min(ssd_hist_trans)}")

        print(f"\tRotation:")
        ssd_degree_rot, relocated_target, orientation_deg = find_min_SSD('rotation', reference, tar,
                                                        vectors=deg, tot_step=total_step)
        save_path_compare_rot = f"output/registration/rotation_MRI1_MRI{2 + image_index}.png"
        plot_degree_compare(reference, relocated_target, ssd_degree_rot, save_path_compare_rot)
        print(f"\t\tOrientation degree: {orientation_deg} degree")
        print(f"\t\tMin SSD: {min(ssd_degree_rot)}\n")

    # - Translation and Rotation with gradient descent function
    gradient_descent(MRI_images[0], MRI_images[1])
