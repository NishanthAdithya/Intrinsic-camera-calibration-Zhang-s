import numpy as np
import cv2
from scipy.optimize import least_squares



def detect_cb_cor(im_path, patt_size) :

    img = cv2.imread(im_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (patt_size[1], patt_size[0]), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        return corners.reshape(-1, 2)
    
    return None


def generate_world_pts(patt_size, sq_size = 21.5) :
    
    r, c = patt_size
    world_pts = np.zeros((r * c, 3), dtype=np.float32)
    
    for i in range(r):
        for j in range(c):
            world_pts[i * c + j] = [j * sq_size, i * sq_size, 0]
    
    return world_pts


def estimate_homo(im_pts, world_pts) :
    
    # extract X, Y from world points zero depth
    X = world_pts[:, 0]
    Y = world_pts[:, 1]
    u = im_pts[:, 0]
    v = im_pts[:, 1]
    
    n = len(im_pts)
    
    # build the A matrix for solving H
    A = np.zeros((2 * n, 9))
    
    for i in range(n):
        A[2*i, :] = [X[i], Y[i], 1, 0, 0, 0, -u[i]*X[i], -u[i]*Y[i], -u[i]]
        A[2*i+1, :] = [0, 0, 0, X[i], Y[i], 1, -v[i]*X[i], -v[i]*Y[i], -v[i]]
    
    # solve using SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape(3, 3)
    
    # normalize the matrix
    H = H / H[2, 2]
    
    return H


def extract_K_from_homo(homographies):
   
    n = len(homographies)
    
    V = []
    
    for H in homographies:
        h1 = H[:, 0]
        h2 = H[:, 1]
        
        # constraitns from each homography
        v12 = np.array([h1[0]*h2[0],h1[0]*h2[1] + h1[1]*h2[0], h1[1]*h2[1], h1[2]*h2[0] + h1[0]*h2[2], h1[2]*h2[1] + h1[1]*h2[2], h1[2]*h2[2]])
        
        v11 = np.array([h1[0]*h1[0], h1[0]*h1[1] + h1[1]*h1[0], h1[1]*h1[1], h1[2]*h1[0] + h1[0]*h1[2], h1[2]*h1[1] + h1[1]*h1[2], h1[2]*h1[2]])
        
        v22 = np.array([h2[0]*h2[0], h2[0]*h2[1] + h2[1]*h2[0], h2[1]*h2[1], h2[2]*h2[0] + h2[0]*h2[2], h2[2]*h2[1] + h2[1]*h2[2], h2[2]*h2[2]])
        
        V.append(v12)
        V.append(v11 - v22)
    
    V = np.array(V)
    
    # solve for b using SVD
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1, :]
    
    B11, B12, B22, B13, B23, B33 = b
    
    denominator = B11 * B22 - B12 * B12
    

    if abs(denominator) < 1e-10:
        
        lambda_val = B33 - B13 * B13 / B11
        if lambda_val / B11 < 0:
            lambda_val = -lambda_val
        
        fx = np.sqrt(lambda_val / B11)
        fy = fx  # square pixels
        cx = -B13 * fx * fx / lambda_val
        cy = B12 * B13 / B11 if abs(B11) > 1e-10 else 0.0
    else:
        v0 = (B12 * B13 - B11 * B23) / denominator
        
        # get lambda
        
        lambda_val = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11
        
        # make sure lambda is positive
        if lambda_val / B11 < 0:
            lambda_val = -lambda_val
        
        # get fx, fy, cx, cy
        fx = np.sqrt(lambda_val / B11)
        fy = np.sqrt(lambda_val * B11 / denominator)
        cx = -B13 * fx * fx / lambda_val
        cy = v0
    
    # K matrix
    gamma = -B12 * fx * fx * fy / lambda_val
    cx = gamma * cy / fy - B13 * fx * fx / lambda_val

    K = np.array([[fx, gamma, cx],[0 , fy   , cy],[0 , 0   , 1 ]])
    
    return K


def extract_Rt(H, K):

   
    K_inv = np.linalg.inv(K)
    
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]
    
    lambda1 = 1.0 / np.linalg.norm(K_inv @ h1)
    lambda2 = 1.0 / np.linalg.norm(K_inv @ h2)
    lambda_avg = (lambda1 + lambda2) / 2.0
    
    # compute r1, r2, r3, t
    r1 = lambda_avg * (K_inv @ h1)
    r2 = lambda_avg * (K_inv @ h2)
    r3 = np.cross(r1, r2) #mutually perpendicular
    t = lambda_avg * (K_inv @ h3)
    
    # form R 
    R = np.column_stack([r1, r2, r3])
    
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    # making sure determinant is 1
    if np.linalg.det(R) < 0:
        R = -R
    
    return R, t


def project_pts(world_pts, K, R, t, k_dist) :

    # convert to camera coordinates
    world_homogeneous = np.column_stack([world_pts, np.ones(len(world_pts))])
    camera_pts = (R @ world_pts.T).T + t
    
    # project to normalized im plane (Essentially were losing scale)
    x = camera_pts[:, 0] / camera_pts[:, 2]
    y = camera_pts[:, 1] / camera_pts[:, 2]
    
    # radial distortion
    r2 = x**2 + y**2
    r4 = r2**2
    k1, k2 = k_dist
    
    x_distorted = x * (1 + k1 * r2 + k2 * r4)
    y_distorted = y * (1 + k1 * r2 + k2 * r4)
    
    # use intrinsic matrix
    points_homogeneous = np.column_stack([x_distorted, y_distorted, np.ones(len(world_pts))])
    im_pts = (K @ points_homogeneous.T).T
    
    # change to inhomogeneous coordinates
    im_pts = im_pts[:, :2] / im_pts[:, 2:3]
    
    return im_pts


def rotation_matrix_to_rodrigues(R) :
    return cv2.Rodrigues(R)[0].flatten()


def rodrigues_to_rotation_matrix(r) :
    return cv2.Rodrigues(r)[0]


def geo_error(params, all_world_pts, all_im_pts):

    fx, fy, cx, cy, gamma, k1, k2 = params[:7]

    K = np.array([
        [fx, gamma, cx],
        [0 , fy   , cy],
        [0 , 0    , 1 ]
    ])

    k_dist = np.array([k1, k2])

    errors = []
    param_idx = 7

    for world_pts, im_pts in zip(all_world_pts, all_im_pts):

        r = params[param_idx:param_idx+3]
        t = params[param_idx+3:param_idx+6]
        param_idx += 6

        R = rodrigues_to_rotation_matrix(r)
        projected = project_pts(world_pts, K, R, t, k_dist)
        error = im_pts - projected
        errors.extend(error.flatten())

    return np.array(errors)


def optimize_calib(initial_K, initial_k, all_world_pts, all_im_pts, all_R, all_t):

    fx = initial_K[0,0]
    fy = initial_K[1,1]
    cx = initial_K[0,2]
    cy = initial_K[1,2]
    gamma = initial_K[0,1]

    k1, k2 = initial_k

    initial_params = [fx, fy, cx, cy, gamma, k1, k2]
    
    for R, t in zip(all_R, all_t):
        r = rotation_matrix_to_rodrigues(R)
        initial_params.extend(r.tolist())
        initial_params.extend(t.tolist())
    
    initial_params = np.array(initial_params)
    
    # optimization using LS 
    result = least_squares(
        geo_error,
        initial_params,
        args=(all_world_pts, all_im_pts),
        method='lm',
        max_nfev=2000
    )
    
    # get optimized parameters
    fx, fy, cx, cy, gamma, k1, k2 = result.x[:7]
    K_opt = np.array([[fx, gamma, cx], [0, fy, cy], [0, 0, 1]])
    k_opt = np.array([k1, k2])
    
    # get optimized R and t for each im
    all_R_opt = []
    all_t_opt = []
    param_idx = 7
    
    for _ in range(len(all_R)):
        r = result.x[param_idx:param_idx+3]
        t = result.x[param_idx+3:param_idx+6]
        param_idx += 6
        
        R_opt = rodrigues_to_rotation_matrix(r)
        all_R_opt.append(R_opt)
        all_t_opt.append(t)
    
    return K_opt, k_opt, all_R_opt, all_t_opt


def reproj_error(K, k_dist,all_world_pts,all_im_pts,all_R, all_t):
    
    total_error = 0.0
    total_pts = 0
    
    for world_pts, im_pts, R, t in zip(all_world_pts, all_im_pts, all_R, all_t):
        projected = project_pts(world_pts, K, R, t, k_dist)
        errors = np.linalg.norm(im_pts - projected, axis=1)
        total_error += np.sum(errors)
        total_pts += len(errors)
    
    return total_error / total_pts


def undistort_im(im, K, k_dist) :
   
    h, w = im.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, np.array([k_dist[0], k_dist[1], 0, 0, 0]), (w, h), 1, (w, h))
    undistorted = cv2.undistort(im, K, np.array([k_dist[0], k_dist[1], 0, 0, 0]), None, new_K)
    return undistorted
