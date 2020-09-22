import math
import numpy as np
import cv2

def calc_n_vector(img_cv, coord):
    height = img_cv.shape[0]
    width = img_cv.shape[1]
    max_x = width - 1
    max_y = height - 1
    x = coord[0]
    y = coord[1]
    latitude = (float(y) / float(max_y) - 0.5) * -180.0
    longitude = (float(x) / float(max_x) - 0.5) * 360.0
    # https://en.wikipedia.org/wiki/N-vector#Converting_latitude/longitude_to_n-vector
    vx = math.cos(math.radians(latitude)) * math.cos(math.radians(longitude))
    vy = math.cos(math.radians(latitude)) * math.sin(math.radians(longitude))
    vz = math.sin(math.radians(latitude))
    # 計算用の座標系から OpenGL の座標系に変換
    vec = np.array([vy, vz, vx])
    nvec = vec / np.linalg.norm(vec)
    return nvec


def calc_coord(img_cv, nvec):
    # n-vector を求める計算の逆。スケーリング係数の導出には不要です。
    height = img_cv.shape[0]
    width = img_cv.shape[1]
    max_x = width - 1
    max_y = height - 1
    # OpenGLの座標系から計算用の座標系に変換
    vx = nvec[2]
    vy = nvec[0]
    vz = nvec[1]
    # https://en.wikipedia.org/wiki/N-vector#Converting_n-vector_to_latitude/longitude
    latitude = math.degrees(math.atan2(vz, math.sqrt((vx ** 2) + (vy ** 2))))
    longitude = math.degrees(math.atan2(vy, vx))
    x = round((longitude / 360.0 + 0.5) * float(max_x))
    y = round((latitude / -180.0 + 0.5) * float(max_y))
    coord = np.array([x, y])
    return coord


def calc_spherical_harmonics_l5(nvec):
    c = np.zeros(36, dtype = float)
    sh = np.zeros(36, dtype = float)

    vec_x = nvec[0]
    vec_y = nvec[1]
    vec_z = nvec[2]

    # L1
    c[ 0] =  0.282095          # Y{0,  0}

    # L2
    c[ 1] = -0.488603          # Y{1, -1}
    c[ 2] =  0.488603          # Y{1,  0}
    c[ 3] = -0.488603          # Y{1,  1}

    # L3
    c[ 4] =  1.092548          # Y{2, -2}
    c[ 5] = -1.092548          # Y{2, -1}
    c[ 6] =  0.315392          # Y{2,  0}
    c[ 7] = -1.092548          # Y{2,  1}
    c[ 8] =  0.546274          # Y{2,  2}

    # L4
    c[ 9] = -0.590044          # Y{3, -3}
    c[10] =  2.89061           # Y{3, -2}
    c[11] = -0.457046          # Y{3, -1}
    c[12] =  0.373176          # Y{3,  0}
    c[13] = -0.457046          # Y{3,  1}
    c[14] =  1.44531           # Y{3,  2}
    c[15] = -0.590044          # Y{3,  3}

    # L5
    c[16] =  2.503343          # Y{4, -4}
    c[17] = -1.770131          # Y{4, -3}
    c[18] =  0.946175          # Y{4, -2}
    c[19] = -0.669047          # Y{4, -1}
    c[20] =  0.105786          # Y{4,  0}
    c[21] = -0.669047          # Y{4,  1}
    c[22] =  0.473087          # Y{4,  2}
    c[23] = -1.770131          # Y{4,  3}
    c[24] =  0.625836          # Y{4,  4}

    c[25] = -0.656383          # Y{5, -5}
    c[26] =  8.302649          # Y{5, -4}
    c[27] = -0.489238          # Y{5, -3}
    c[28] =  4.793537          # Y{5, -2}
    c[29] = -0.452947          # Y{5, -1}
    c[30] =  0.116950          # Y{5,  0}
    c[31] = -0.452947          # Y{5,  1}
    c[32] =  2.396768          # Y{5,  2}
    c[33] = -0.489238          # Y{5,  3}
    c[34] =  2.075662          # Y{5,  4}
    c[35] = -0.656383          # Y{5,  5}

    # Y{0,0} 
    sh[ 0] = c[ 0]

    # Y{1,-1}, Y{1,0}, Y{1,1}
    sh[ 1] = c[ 1] * vec_y
    sh[ 2] = c[ 2] * vec_z
    sh[ 3] = c[ 3] * vec_x

    # Y{2, -2}, Y{2,-1}, Y{2,1}
    sh[ 4] = c[ 4] * vec_x * vec_y
    sh[ 5] = c[ 5] * vec_y * vec_z
    sh[ 7] = c[ 7] * vec_x * vec_z

    # Y{2,0} 
    sh[ 6] = c[ 6] * ( 3.0 * vec_z * vec_z - 1.0 )

    # Y{2,2} 
    sh[ 8] = c[ 8] * ( vec_x * vec_x - vec_y * vec_y )

    # Y{3, -3} = A * sqrt(5/8) * (3 * x^2 * y - y^3)
    sh[ 9] = c[ 9] * ( 3.0 * vec_x * vec_x * vec_y - vec_y * vec_y * vec_y ) 

    # Y{3, -2} = A * sqrt(15) * x * y * z 
    sh[10] = c[10] * vec_x * vec_y * vec_z

    # Y{3, -1} = A * sqrt(3/8) * y * (5 * z^2 - 1)
    sh[11] = c[11] * vec_y * ( 5.0 * vec_z * vec_z - 1.0 )

    # Y{3,  0} = A * (1/2) * (5 * z^3 - 3 *z)	
    sh[12] = c[12] * ( 5.0 * vec_z * vec_z * vec_z - 3.0 * vec_z )

    # Y{3,  1} = A * sqrt(3/8) * x * (5 * z^2 - 1)
    sh[13] = c[13] * vec_x * ( 5.0 * vec_z * vec_z  - 1.0 )

    # Y{3,  2} = A * sqrt(15/4) * z *(x^2 - y^2)
    sh[14] = c[14] * vec_z * ( vec_x * vec_x - vec_y * vec_y )

    # Y{3,  3} = A * sqrt(5/8) * (x^3 - 3 * x * y^2)
    sh[15] = c[15] * ( vec_x * vec_x * vec_x - 3.0 * vec_x * vec_y * vec_y )

    x2 = vec_x * vec_x
    y2 = vec_y * vec_y
    z2 = vec_z * vec_z
    x4 = x2 * x2
    y4 = y2 * y2
    z4 = z2 * z2
    sh[16] = c[16] * vec_y * vec_x * ( x2 - y2 )               # 4, -4
    sh[17] = c[17] * vec_y * ( 3.0 * x2 - y2 ) * vec_z        # 4, -3
    sh[18] = c[18] * vec_y * vec_x * ( -1.0 + 7.0 * z2 )     # 4, -2
    sh[19] = c[19] * vec_y * vec_z * ( -3.0 + 7.0 * z2 )     # 4, -1
    sh[20] = c[20] * ( 35.0 * z4 - 30.0 * z2 + 3.0 )        # 4, 0
    sh[21] = c[21] * vec_x * vec_z * ( -3.0 + 7.0 * z2 )     # 4, 1
    sh[22] = c[22] * ( x2 - y2 ) * ( -1.0 + 7.0 * z2 )       # 4, 2
    sh[23] = c[23] * vec_x * ( x2 - 3.0 * y2 ) * vec_z        # 4, 3
    sh[24] = c[24] * ( x4 - 6.0 * y2 * x2 + y4 )              # 4, 4

    sh[25] = c[25] * vec_y * ( 5.0 * x4 - 10.0 * y2 * x2 + y4 )          # 5, -5
    sh[26] = c[26] * vec_y * vec_x * ( x2 - y2 ) * vec_z                   # 5, -4
    sh[27] = c[27] * vec_y * ( 3.0 * x2 - y2 ) * ( -1.0 + 9.0 * z2 )    # 5, -3
    sh[28] = c[28] * vec_y * vec_x * vec_z * ( -1.0 + 3.0 * z2 )         # 5, -2
    sh[29] = c[29] * vec_y * ( -14.0 * z2 + 21.0 * z4 + 1.0 )           # 5, -1
    sh[30] = c[30] * vec_z * ( 63.0 * z4 - 70.0 * z2 + 15.0 )           # 5, 0
    sh[31] = c[31] * vec_x * ( -14.0 * z2 + 21.0 * z4 + 1.0 )           # 5, 1
    sh[32] = c[32] * ( x2 - y2 ) * vec_z * ( -1.0 + 3.0 * z2 )           # 5, 2
    sh[33] = c[33] * vec_x * ( x2 - 3.0 * y2 ) * ( -1.0 + 9.0 * z2 )    # 5, 3
    sh[34] = c[34] * ( x4 - 6.0 * y2 * x2 + y4 ) * vec_z                  # 5, 4
    sh[35] = c[35] * vec_x * ( x4 - 10.0 * y2 * x2 + 5.0 * y4 )          # 5, 5

    return sh


def calc_solid_angle(img_cv, coord):
    height = img_cv.shape[0]
    width = img_cv.shape[1]
    x = coord[0]
    y = coord[1]
    # https://en.wikipedia.org/wiki/Solid_angle#Latitude-longitude_rectangle
    phi_n = math.radians((float(y) / float(height) - 0.5) * -180.0)
    phi_s = math.radians((float(y + 1) / float(height) - 0.5) * -180.0)
    theta_e = math.radians((float(x + 1) / float(width) - 0.5) * 360.0)
    theta_w = math.radians((float(x) / float(width) - 0.5) * 360.0)
    dw = (math.sin(phi_n) - math.sin(phi_s)) * (theta_e - theta_w)
    return dw


def test():

    # 緯度経度マップ HDR 読み込み（左上原点）
    file_name = "EpicQuadPanorama_CC+EV1.HDR"
    img_cv = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

    # おそいので小さくしておく
    img_cv = cv2.resize(img_cv , (int(img_cv.shape[1] / 4), int(img_cv.shape[0] / 4)))

    #型情報をダンプ
    print(type(img_cv))
    print(type(img_cv.shape))
    print(img_cv.shape)

    # 画像の大きさを取得
    height, width, channels = img_cv.shape[:3]
    print("height: " + str(height))
    print("width: " + str(width))
    print("channels: " + str(channels))
    print("type：" + str(img_cv.dtype))

    cv2.imshow("img_cv", img_cv)
    #cv2.waitKey()

    # 法線マップ表示
    nmap = np.zeros((height, width, channels), dtype = float)
    for y in range(height):
        for x in range(width):
            coord = np.array([x, y])
            nvec = calc_n_vector(img_cv, coord)
            #coord2 = calc_coord(img_cv2, nvec)
            #if coord[0] != coord2[0] or coord[1] != coord2[1]:
            #   continue
            pixel = np.zeros(3, dtype = float)
            # BGR
            pixel[0] = nvec[2] / 2.0 + 0.5
            pixel[1] = nvec[1] / 2.0 + 0.5
            pixel[2] = nvec[0] / 2.0 + 0.5
            nmap[y, x] = pixel
    cv2.imshow("nmap", nmap)
    #cv2.waitKey()

    # スケーリング係数を求める 
    coeff = np.zeros((36, 3), dtype = float)
    dbg_dw = 0.0
    for y in range(height):
        for x in range(width):
            coord = np.array([x, y])
            nvec = calc_n_vector(img_cv, coord)
            sh = calc_spherical_harmonics_l5(nvec)
            dw = calc_solid_angle(img_cv, coord)
            dbg_dw = dbg_dw + dw
            pixel = img_cv[y,x]
            for i in range(36):
                coeff[i][0] += pixel[0] * sh[i] * dw
                coeff[i][1] += pixel[1] * sh[i] * dw
                coeff[i][2] += pixel[2] * sh[i] * dw
    print("scaling coefficient = " + str(coeff))
    print("solid angle (4π) = " + str(4 * math.pi))
    print("solid angle dw total = " + str(dbg_dw))

    # 展開 
    img_cv2 = np.zeros((height, width, channels), dtype = float)
    for y in range(height):
        for x in range(width):
            coord = np.array([x, y])
            nvec = calc_n_vector(img_cv2, coord)
            sh = calc_spherical_harmonics_l5(nvec)
            pixel = np.zeros(3, dtype = float)
            for i in range(36):
                pixel[0] += coeff[i][0] * sh[i]
                pixel[1] += coeff[i][1] * sh[i]
                pixel[2] += coeff[i][2] * sh[i]
            img_cv2[y, x] = pixel
    cv2.imshow("img_cv2", img_cv2)

    cv2.waitKey() 

if __name__ == '__main__':
    test()
