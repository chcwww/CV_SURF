import numpy as np
import cv2
import functools
"""
Keep the result of function in cache, when getting same input 
we can use the result in the cache directly, and the cache is
using LRU replacement algorithm.
"""
from functools import lru_cache
import time 

# Like a timer
def decorator(func) :
    @functools.wraps(func)
    def wrapper(*args, **kwargs) :
        st = time.time()
        print(f"{func.__name__} running...")
        result = func(*args, **kwargs)
        en = time.time()
        print(f"Time : {en - st:.2f} seconds")
        # if(en - st > threshold) :
        #     print("The program is toooo slow...")
        return result
    return wrapper


class SURF:
    def __init__(self, octaves=3, layers=4, threshold=1000):
        self.octaves = octaves  # 幾組
        self.layers = layers # 金字塔高度
        self.threshold = threshold # for Hessian
        self.params_related_to_L = {}
        self.octaves_layers_L = {}

        # 這裡的L其實都是論文的l0 (L/3) 以下論文的L用L_P代替
        for o in range(1, self.octaves + 1):
            self.octaves_layers_L[o] = {}
            for i in range(1, self.layers + 1):
                L = (2 ** o) * i + 1 # 2, 4, 8 ... 來算每一組的尺寸(l0)
                self.octaves_layers_L[o][i] = L
                # 每個L所對應的一些參數 
                # w就不算了直接當0.9
                self.params_related_to_L[L] = {
                    'sigma_L': round(0.4 * L, 2), # sigma = 1.2 * L_P/9 = 0.4 * l0 
                    'L': L, # l0
                    'l': np.int(0.8 * L) # descip sigma
                }

        L_max = 3 * ((2 ** self.octaves) * self.layers + 1)
        self.border_size = L_max

    @decorator
    def init(self, gray_image: np.ndarray):
        self.image = gray_image
        self.build_stretched_image()
        # self.image_broadcast = self.image
        self.construct_image_integral()
        self.compute_determinant_of_hessian()

    @decorator
    def detectAndCompute(self):
        """
        Returns list of keypoints cv2.KeyPoint and SURF descriptors
        Each descriptor is a list with 3 elements:
         - (64,) descriptor vector
         - orientation
         - sign of Laplassian
        """
        key_points = self.keypoints_selection()

        cv2_key_points = []
        descriptors = []

        for key_point in key_points:
            x, y, L, orientation = key_point
            point = cv2.KeyPoint(
                x = x,
                y = y,
                # size=self.get_box_space_parameter_by_L(L, 'sigma_L'),
                size  = L,
                angle = orientation * 180 / np.pi
            )
            cv2_key_points.append(point)

            descriptor = self.compute_point_descriptor(key_point)
            descriptors.append(list(descriptor))

        return cv2_key_points, descriptors

    def construct_image_integral(self):
        # I(x, y) = sum_i^x sum_j^y f(i, j)
        self.image_integral = np.cumsum(np.cumsum(self.image_broadcast, axis=0), axis=1)
        self.image_integral = self.image_broadcast.copy()

        for i in range(self.image_broadcast.shape[0]) :
            for j in range(self.image_broadcast.shape[1]) :
                self.image_integral[i, j] = self.image_broadcast[i, j] \
                + (self.image_integral[i, j-1] if j-1 > -1 else 0) \
                + (self.image_integral[i-1, j] if i-1 > -1 else 0) \
                - (self.image_integral[i-1, j-1] if i-1 > -1 and j-1 > -1 else 0)


    def build_stretched_image(self):
        """
        Original image will be extended with border size n around
        and this border will be filled by mirror of the original image
        """
        n = self.border_size
        h = self.image.shape[0]
        w = self.image.shape[1]

        self.image_broadcast = np.ndarray((h + 2 * n, w + 2 * n))
        self.image_broadcast[n:n + h, n:n + w] = self.image
        self.image_broadcast[0:n, n:n + w] = np.flipud(self.image[0:n, :])
        self.image_broadcast[n + h:, n:n + w:] = np.flipud(self.image[h - n:, :])
        self.image_broadcast[:, 0:n] = np.fliplr(self.image_broadcast[:, n:2 * n])
        self.image_broadcast[:, -n:] = np.fliplr(self.image_broadcast[:, -2 * n:-n])

    @lru_cache
    def get_box_space_parameter_by_L(self, L: int, param: str):
        return self.params_related_to_L[L][param]

    def box_filter(self, a: int, b: int, c: int, d: int, x: int, y: int):
        """
        Consider params as rectangular domain: [a,b]x[c,d] - not points!!
        and point (x,y)
        """

        xa = self.border_size + x - a - 1
        yc = self.border_size + y - c - 1
        xb = self.border_size + x - b - 1 - 1
        yd = self.border_size + y - d - 1 - 1

        cons_xa = (0 <= xa < self.image_integral.shape[1])
        cons_xb = (0 <= xb < self.image_integral.shape[1])
        cons_yc = (0 < yc < self.image_integral.shape[0])
        cons_yd = (0 < yd < self.image_integral.shape[0])
        
        u1 = self.image_integral[yc, xa] if cons_xa and cons_yc else 0
        u2 = self.image_integral[yd, xb] if cons_xb and cons_yd else 0
        u3 = self.image_integral[yd, xa] if cons_xa and cons_yd else 0
        u4 = self.image_integral[yc, xb] if cons_xb and cons_yc else 0

        return u1 + u2 - u3 - u4

    def get_image_value(self, x: int, y: int):
        x_is = self.border_size + x
        y_is = self.border_size + y

        return self.image_broadcast[y_is][x_is]

    def compute_Dx_Haar(self, l: int, x: int, y: int):
        return self.box_filter(-l, -1, -l, l, x, y) - self.box_filter(1, l, -l, l, x, y)

    def compute_Dy_Haar(self, l: int, x: int, y: int):
        return self.box_filter(-l, l, -l, -1, x, y) - self.box_filter(-l, l, 1, l, x, y)

    def compute_Dxx(self, L: int, x: int, y: int):
        (b1, b2), (b3, b4), (b5, b6), (b7, b8) = self.get_Dxx_L_domains(L)
        B1 = self.box_filter(b1, b2, b3, b4, x, y)
        B2 = self.box_filter(b5, b6, b7, b8, x, y)
        return B1 - 3 * B2

    def compute_Dyy(self, L: int, x: int, y: int):
        (b1, b2), (b3, b4), (b5, b6), (b7, b8) = self.get_Dyy_L_domains(L)
        B1 = self.box_filter(b1, b2, b3, b4, x, y)
        B2 = self.box_filter(b5, b6, b7, b8, x, y)
        return B1 - 3 * B2

    def compute_Dxy(self, L: int, x: int, y: int):
        (ne1, ne2), (ne3, ne4), \
            (nw1, nw2), (nw3, nw4), \
            (sw1, sw2), (sw3, sw4), \
            (se1, se2), (se3, se4) \
            = self.get_Dxy_L_domains(L)

        north_east_quadrant = self.box_filter(ne1, ne2, ne3, ne4, x, y)
        north_west_quadrant = self.box_filter(nw1, nw2, nw3, nw4, x, y)
        south_west_quadrant = self.box_filter(sw1, sw2, sw3, sw4, x, y)
        south_east_quadrant = self.box_filter(se1, se2, se3, se4, x, y)

        return north_east_quadrant + south_west_quadrant - north_west_quadrant - south_east_quadrant

    @lru_cache
    def get_Dxx_L_domains(self, L: int):
        b1 = np.int((L * 3 - 1) / 2)
        b2 = np.int((L - 1) / 2)
        return [
            (-b1, b1),
            (-(L - 1), (L - 1)),
            (-b2, b2),
            (-(L - 1), (L - 1))
        ]

    @lru_cache
    def get_Dyy_L_domains(self, L: int):
        b1 = np.int((L * 3 - 1) / 2)
        b2 = np.int((L - 1) / 2)
        return [
            (-(L - 1), (L - 1)),
            (-b1, b1),
            (-(L - 1), (L - 1)),
            (-b2, b2)
        ]

    @lru_cache
    def get_Dxy_L_domains(self, L: int):
        return [
            (1, L), (1, L),  # ++
            (-L, -1), (1, L),  # -+
            (-L, -1), (-L, -1),  # --
            (1, L), (-L, -1),  # +-
        ]

    @lru_cache
    def get_L_index_by_L(self, L: int):
        if L not in self.params_related_to_L:
            return False

        return list(self.params_related_to_L).index(L)

    @lru_cache
    def get_L_by_L_index(self, L_index: int):
        if 0 <= L_index < len(self.params_related_to_L):
            list_L = list(self.params_related_to_L)
            return list_L[L_index]

        return False

    def compute_determinant_of_hessian(self):
        self.getDH.cache_clear()
        self.getSignLaplassian.cache_clear()

        used_L = set({})
        self.DoH = np.ndarray(
            (len(self.params_related_to_L),
             self.image.shape[0],
             self.image.shape[1])
        )
        self.lapSign = np.ndarray(
            (len(self.params_related_to_L),
             self.image.shape[0],
             self.image.shape[1])
        )
        DoH_index = 0

        for o in range(1, self.octaves + 1):
            step = 2 ** (o - 1) # 一次算多大塊?

            for i in range(1, self.layers + 1):
                L = (2 ** o) * i + 1

                if L in used_L:
                    continue # 算過了

                used_L.add(L)

                normalizer = 1 / (L ** 4)
                w = 0.9 # 直接用0.9
                x = 0

                DoH_L = np.ndarray(self.image.shape)
                lapSign_L = np.ndarray(self.image.shape)

                while x < self.image.shape[1]:
                    y = 0
                    while y < self.image.shape[0]:
                        # 算高斯 (以Box Filter來逼近)
                        Dxx = self.compute_Dxx(L, x, y)
                        Dyy = self.compute_Dyy(L, x, y)
                        Dxy = self.compute_Dxy(L, x, y)
                        
                        # 其approximate的DoH (Determinant of Hessian)
                        DoH = normalizer * (Dxx * Dyy - ((w * Dxy) ** 2))
                        DoH_L[y][x] = DoH

                        lapSign_L[y][x] = 1 if (Dxx + Dyy) > 0 else -1

                        y += step

                    x += step

                self.DoH[DoH_index] = DoH_L
                self.lapSign[DoH_index] = lapSign_L
                DoH_index += 1

    @lru_cache
    def getDH(self, L: int, x: int, y: int):
        exist = (0 <= L < self.DoH.shape[0]) \
                and (0 <= x < self.DoH.shape[2]) \
                and (0 <= y < self.DoH.shape[1])

        if not exist:
            return -np.inf

        val = self.DoH[L][y][x]

        return val

    @lru_cache
    def getSignLaplassian(self, L: int, x: int, y: int):
        exist = (0 <= L < self.lapSign.shape[0]) \
                and (0 <= x < self.lapSign.shape[2]) \
                and (0 <= y < self.lapSign.shape[1])

        if not exist:
            return 0

        val = self.lapSign[L][y][x]

        return val

    def keypoints_selection(self):
        keyPoints = []
        reffal = 0

        for o in range(1, self.octaves + 1):
            step = 2 ** (o - 1)

            for i in range(2, np.min([self.layers, 4])):
                L = (2 ** o) * i + 1
                L_index = self.get_L_index_by_L(L) # 這個l0在param那邊是第幾個(因為有重疊的)
                x = 0

                while x < self.image.shape[1]:
                    y = 0
                    while y < self.image.shape[0]:
                        DoH = self.getDH(L_index, x, y) # 拿回Hessian值

                        if DoH > self.threshold:
                            if self.local_extrema_DoH_bool(L_index, x, y, DoH):
                                is_re_keypoint, point = self.is_refined(x, y, L, o)

                                if is_re_keypoint and point[2] is False:
                                    reffal += 1
                                if is_re_keypoint and point[2] is not False:
                                    key_point_with_orientation = self.compute_key_point_orientation(point)
                                    keyPoints.append(key_point_with_orientation)

                        y += step

                    x += step

        # print(f'Refined and false L {reffal}')

        return keyPoints

    def local_extrema_DoH_bool(self, L, x, y, current_DoH):
        lL = L - 1 if L - 1 >= 0 else 0
        lx = x - 1 if x - 1 >= 0 else 0
        ly = y - 1 if y - 1 >= 0 else 0

        try:
            max_value = np.amax(self.DoH[lL:L + 2, ly:y + 2, lx:x + 2])
        except BaseException:
            max_value = np.inf
            pass

        if max_value > current_DoH:
            return False

        return True

    def is_refined(self, x0, y0, L, o):
        p = 2 ** (o - 1)
        p2 = p ** 2
        L0 = self.get_L_index_by_L(L)
        Lp2p = self.get_L_index_by_L(L + 2 * p)
        Lm2p = self.get_L_index_by_L(L - 2 * p)

        Hxx = (self.getDH(L0, x0 + p, y0) + self.getDH(L0, x0 - p, y0)
               - 2 * self.getDH(L0, x0, y0)) / p2

        Hyy = (self.getDH(L0, x0, y0 + p) + self.getDH(L0, x0, y0 - p)
               - 2 * self.getDH(L0, x0, y0)) / p2

        Hxy = (self.getDH(L0, x0 + p, y0 + p) + self.getDH(L0, x0 - p, y0 - p)
               - self.getDH(L0, x0 - p, y0 + p) - self.getDH(L0, x0 + p, y0 - p)) / (4 * p2)

        HxL = (self.getDH(Lp2p, x0 + p, y0) + self.getDH(Lm2p, x0 - p, y0)
               - self.getDH(Lp2p, x0 - p, y0) - self.getDH(Lm2p, x0 + p, y0)) / (8 * p2)

        HyL = (self.getDH(Lp2p, x0, y0 + p) + self.getDH(Lm2p, x0, y0 - p)
               - self.getDH(Lp2p, x0, y0 - p) - self.getDH(Lm2p, x0, y0 + p)) / (8 * p2)

        HLL = (self.getDH(Lp2p, x0, y0) + self.getDH(Lm2p, x0, y0) - 2 * self.getDH(L0, x0, y0)) / (4 * p2)

        dx = (self.getDH(L0, x0 + p, y0) - self.getDH(L0, x0 - p, y0)) / (2 * p)
        dy = (self.getDH(L0, x0, y0 + p) - self.getDH(L0, x0, y0 - p)) / (2 * p)
        dL = (self.getDH(Lp2p, x0, y0) - self.getDH(Lm2p, x0, y0)) / (4 * p)

        # build matrix H0
        H0 = np.zeros((3, 3))
        np.fill_diagonal(H0, [Hxx, Hyy, HLL])
        H0[0, 1] = H0[1, 0] = Hxy
        H0[0, 2] = H0[2, 0] = HxL
        H0[1, 2] = H0[2, 1] = HyL

        # build vector d0
        d0 = np.array([dx, dy, dL])

        # compute ksi
        try:
            ksi = np.dot(np.linalg.inv(H0), d0)

            if np.max(np.abs(ksi * [1, 1, 0.5])) < p:
                x = np.int(x0 + ksi[0])
                y = np.int(y0 + ksi[1])
                L_index = np.int(L0 + ksi[2])

                return True, (x, y, self.get_L_by_L_index(L_index))
        except np.linalg.LinAlgError:
            pass

        return False, (-1, -1, -1)

    @lru_cache
    def get_gaussian(self, x, y, sigma):
        s2 = 2 * sigma * sigma
        return np.exp(-(x * x + y * y) / s2) / (np.pi * s2)

    def compute_key_point_orientation(self, key_point):
        x0, y0, L = key_point
        l = self.get_box_space_parameter_by_L(L, 'l')
        sigma = self.get_box_space_parameter_by_L(L, 'sigma_L')
        sectors = 20

        list_fi = np.ndarray((0, 3))
        list_theta = np.ndarray((0, 4))
        G_sum = 0
        G_list = {}

        for i in range(-6, 7):
            for j in range(-6, 7):
                if i * i + j * j <= 36:
                    x = np.int(x0 + i * sigma)
                    y = np.int(y0 + j * sigma)
                    G = self.get_gaussian(x - x0, y - y0, 2 * sigma)
                    G_sum += G
                    G_list.setdefault(10 * i + j, G)

        for i in range(-6, 7):
            for j in range(-6, 7):
                if i * i + j * j <= 36:
                    x = np.int(x0 + i * sigma)
                    y = np.int(y0 + j * sigma)
                    DxL = self.compute_Dx_Haar(l, x, y)
                    DyL = self.compute_Dy_Haar(l, x, y)
                    G = G_list.get(10 * i + j) / G_sum
                    image_value = self.get_image_value(x, y)

                    fi = np.array([[
                        np.arctan2(DyL, DxL),
                        DxL * image_value * G,
                        DyL * image_value * G,
                    ]])

                    list_fi = np.concatenate([list_fi, fi])

        for k in range(0, sectors * 2):
            theta = k * np.pi / sectors

            mask = (theta - np.pi / 6 <= list_fi[:, 0]) & (list_fi[:, 0] <= theta + np.pi / 6)
            FI_theta = np.sum(list_fi[mask, 1:], axis=0)
            FI_theta_norm = np.linalg.norm(FI_theta)

            list_theta = np.concatenate([
                list_theta,
                np.concatenate([[theta], [FI_theta_norm], FI_theta]).reshape(1, 4)
            ])

        max_theta_norm_index = np.argmax(list_theta[:, 1])
        orientation = np.arctan2(list_theta[max_theta_norm_index][3], list_theta[max_theta_norm_index][2])

        return x0, y0, L, orientation

    def compute_point_descriptor(self, point):
        x0, y0, L, theta = point
        Dl = self.get_box_space_parameter_by_L(L, 'l')
        sigma = self.get_box_space_parameter_by_L(L, 'sigma_L')
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        R_minus = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta), np.cos(-theta)],
        ])

        G_sum = 0
        G_list = {}
        step = 1

        # compute gausians to be able to get normalized discrete gausian
        for i in range(0, 4):
            for j in range(0, 4):
                for k in range(0, 5):
                    for l in range(0, 5):
                        u = (i - 2) * 5 + k + 0.5
                        v = (j - 2) * 5 + l + 0.5

                        sd = 3.3 * sigma
                        G = self.get_gaussian(k, l, sd)
                        G_sum += G
                        G_list.setdefault(1000 * k + l, G)

        mk = np.zeros((4, 4, 4))

        for i in range(0, 4):
            for j in range(0, 4):
                sum_dx_uv = 0
                sum_dy_uv = 0
                sum_dx_uv_abs = 0
                sum_dy_uv_abs = 0

                for k in range(0, 5):
                    for l in range(0, 5):
                        u = (i - 2) * 5 + k + 0.5
                        v = (j - 2) * 5 + l + 0.5

                        x, y = sigma * (R @ np.array([u, v])) + np.array([x0, y0])
                        x = np.int(x)
                        y = np.int(y)

                        try:
                            DxL = self.compute_Dx_Haar(Dl, x, y)
                            DyL = self.compute_Dy_Haar(Dl, x, y)
                        except BaseException as e:
                            # print(x0, y0, L, theta, e)
                            pass

                        G = G_list.get(1000 * k + l) / G_sum

                        dx_uv, dy_uv = (R_minus @ np.array([DxL, DyL])) * G

                        sum_dx_uv += dx_uv
                        sum_dy_uv += dy_uv
                        sum_dx_uv_abs += np.abs(dx_uv)
                        sum_dy_uv_abs += np.abs(dy_uv)

                mk[i, j, 0] += sum_dx_uv
                mk[i, j, 1] += sum_dy_uv
                mk[i, j, 2] += sum_dx_uv_abs
                mk[i, j, 3] += sum_dy_uv_abs

        mk = mk.reshape((64,))

        norm = np.linalg.norm(mk)

        if norm != 0:
            mk = mk / norm

        return mk, theta, self.getSignLaplassian(L, x0, y0)

class SURF_Matcher():
    def match(self, descriptors1, descriptors2, threshold=0.75):
        """
        Returns list of descriptors [cv2.DMatch]
        According to SURF match logic only distance between descriptor vectors with
        the same sign of Laplassian are considered
        """
        matches = []

        v1_index = 0
        for desc_vector1, theta1, SignLaplassian1 in descriptors1:
            lengths_to_point = np.ndarray((len(descriptors2),))
            v2_index = 0

            for desc_vector2, theta2, SignLaplassian2 in descriptors2:
                dist = np.inf

                if (SignLaplassian1 == SignLaplassian2):
                    # 兩vector的距離
                    dist = np.linalg.norm(desc_vector1 - desc_vector2)

                lengths_to_point[v2_index] = dist
                v2_index += 1

            indxs_sorted = np.argpartition(lengths_to_point, 2) # 得到距離最短兩個的index
            dist1 = lengths_to_point[indxs_sorted[0]] # min
            dist2 = lengths_to_point[indxs_sorted[1]] # second min

            if dist2 == 0 or dist1 / dist2 <= threshold: # 第一個足夠的小
                match = cv2.DMatch(
                    _distance = dist1, # 距離
                    _queryIdx = v1_index, # 在原始的idx
                    _trainIdx = indxs_sorted[0] # 在新的比較圖的idx
                )

                matches.append(match)

            v1_index += 1

        return matches