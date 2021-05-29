import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import datetime as dt
import os


def f2i(f):
	return int(round(f))


def set_range(v, min, max):
	# while v < min:
	# 	v += (max - min)
	#
	# while v >= max:
	# 	v -= (max - min)

	return v

def waku():
	outp = 'output/waku/'
	os.makedirs(outp, exist_ok=True)
	print(dt.datetime.now())
	img = cv2.cvtColor(cv2.imread('input/minato.jpg', 1), cv2.COLOR_BGR2RGB)
	lss = [[137,230,1052,1165],[380,466,100,220],[f2i(img.shape[0] * 0.8),f2i(img.shape[0] * 0.9),f2i(img.shape[1] * 0.2),f2i(img.shape[1] * 0.3)],[f2i(img.shape[0] * 0.1),f2i(img.shape[0] * 0.3), f2i(img.shape[1] * 0.1),f2i(img.shape[1] * 0.2)]]
	im_save_waku(img, outp + 'waku.png',lss)

def main():
	outp = 'output/full/'
	os.makedirs(outp, exist_ok=True)
	print(dt.datetime.now())
	img = cv2.cvtColor(cv2.imread('input/minato.jpg', 1), cv2.COLOR_BGR2RGB)
	# img = img[f2i(img.shape[0] * 0.1):f2i(img.shape[0] * 0.3), f2i(img.shape[1] * 0.1):f2i(img.shape[1] * 0.2)]
	# img = img[f2i(img.shape[0] * 0.8):f2i(img.shape[0] * 0.9), f2i(img.shape[1] * 0.2):f2i(img.shape[1] * 0.3)]  # 色えぐい
	# img = img[137:230, 1052:1165]  # ビル
	# img = img[380:466, 100:220] # 船の帆
	im_save(img, outp + 'original.png')
	out0 = rgb2bayer(img, 0)
	out1 = rgb2bayer(img, 1)
	im_save(out0, outp + 'bayer.png')
	im_save(out1, outp + 'bayer_rgb.png')
	dst = simple_func(out1)
	print(dt.datetime.now())
	# dst2 = acpi_func(out0)
	print(dt.datetime.now())
	im_save(dst, outp + 'simple.png')


def simple_func(src):
	dst = np.zeros(src.shape)
	for (i, ch) in enumerate(['R', 'G', 'B']):
		if ch == 'G':
			karnel = np.array([[0, 0.25, 0],
			                   [0.25, 1, 0.25],
			                   [0, 0.25, 0]])
		else:
			karnel = np.array([[0.25, 0.5, 0.25],
			                   [0.5, 1, 0.5],
			                   [0.25, 0.5, 0.25]])

		dst[:, :, i] = cv2.filter2D(src[:, :, i], -1, karnel)
	return dst


def hokan_func(src):
	print(src.shape)
	src = np.insert(src, 0, 0, axis=0)
	src = np.insert(src, 0, 0, axis=0)
	src = np.insert(src, -1, 0, axis=0)
	src = np.insert(src, -1, 0, axis=0)
	src = np.insert(src, 0, 0, axis=1)
	src = np.insert(src, 0, 0, axis=1)
	src = np.insert(src, -1, 0, axis=1)
	src = np.insert(src, -1, 0, axis=1)
	print(src.shape)

	dst = np.zeros(src.shape + (3,))

	for i in range(2, src.shape[0] - 2):
		for j in range(2, src.shape[1] - 2):
			if np.mod(i + j, 2) == 0:  # R or B
				alp = abs((src[i - 2, j] + src[i + 2, j]) / 2 - src[i, j])
				bet = abs((src[i, j - 2] + src[i, j + 2]) / 2 - src[i, j])
				if alp == bet:
					gout = (src[i - 1, j] + src[i + 1, j] + src[i, j - 1] + src[i, j + 1]) / 4
				else:
					if np.mod(i, 2) < 2:  # R
						if alp < bet:
							gout = (src[i - 1, j] + src[i + 1, j]) / 2
						elif alp > bet:
							gout = (src[i, j - 1] + src[i, j + 1]) / 2
					else:
						if alp > bet:
							gout = (src[i - 1, j] + src[i + 1, j]) / 2
						elif alp < bet:
							gout = (src[i, j - 1] + src[i, j + 1]) / 2

				dst[i, j, 1] = gout
			else:  # G

				dst[i, j, 1] = src[i, j]
	print(dst.max(), dst.min())
	print(src.max(), src.min())
	n = 0
	for i in range(2, src.shape[0] - 2):
		for j in range(2, src.shape[1] - 2):
			if np.mod(i, 2) == 0:
				if np.mod(j, 2) == 0:
					pos = (0, 0)  # 'R'
					dst[i, j, 0] = src[i, j]
					dst[i, j, 2] = ((src[i - 1, j - 1] - dst[i - 1, j - 1, 1]) + (src[i - 1, j + 1] - dst[
						i - 1, j + 1, 1]) + (src[i + 1, j - 1] - dst[i + 1, j - 1, 1]) + (src[i + 1, j + 1] - dst[
						i + 1, j + 1, 1])) / 4 + dst[i, j, 1]

				else:
					pos = (0, 1)  # G
					dst[i, j, 0] = ((src[i, j - 1] - dst[i, j - 1, 1]) + (src[i, j + 1] - dst[i, j + 1, 1])) / 2 + dst[
						i, j, 1]
					dst[i, j, 2] = ((src[i - 1, j] - dst[i - 1, j, 1]) + (src[i + 1, j] - dst[i + 1, j, 1])) / 2 + dst[
						i, j, 1]
					v = dst[i, j, 2]
					if v < 0 or v > 256:
						n += 1
						print(v)
			else:
				if np.mod(j, 2) == 0:
					pos = (1, 0)  # G
					dst[i, j, 0] = ((src[i - 1, j] - dst[i - 1, j, 1]) + (src[i + 1, j] - dst[i + 1, j, 1])) / 2 + dst[
						i, j, 1]
					dst[i, j, 2] = ((src[i, j - 1] - dst[i, j - 1, 1]) + (src[i, j + 1] - dst[i, j + 1, 1])) / 2 + dst[
						i, j, 1]
				else:
					pos = (1, 1)  # B
					dst[i, j, 0] = ((src[i - 1, j - 1] - dst[i - 1, j - 1, 1]) + (
							src[i - 1, j + 1] - dst[i - 1, j + 1, 1]) + (src[i + 1, j - 1] - dst[i + 1, j - 1, 1]) + (
							                src[i + 1, j + 1] - dst[i + 1, j + 1, 1])) / 4 + dst[i, j, 1]
					dst[i, j, 2] = src[i, j]
	print('n={}'.format(n))
	print(dst.max(), dst.min())
	# dst = (dst - dst.min()) * 255 / (dst.max()-dst.min())
	dst = np.delete(dst, 0, axis=0)
	dst = np.delete(dst, 0, axis=0)
	dst = np.delete(dst, 0, axis=0)
	dst = np.delete(dst, -1, axis=0)
	dst = np.delete(dst, -1, axis=0)
	dst = np.delete(dst, -1, axis=0)
	dst = np.delete(dst, 0, axis=1)
	dst = np.delete(dst, 0, axis=1)
	dst = np.delete(dst, 0, axis=1)
	dst = np.delete(dst, -1, axis=1)
	dst = np.delete(dst, -1, axis=1)
	dst = np.delete(dst, -1, axis=1)
	print(dst.shape)

	return dst


def acpi_func(src):
	print(src.shape)
	src = np.insert(src, 0, 0, axis=0)
	src = np.insert(src, 0, 0, axis=0)
	src = np.insert(src, -1, 0, axis=0)
	src = np.insert(src, -1, 0, axis=0)
	src = np.insert(src, 0, 0, axis=1)
	src = np.insert(src, 0, 0, axis=1)
	src = np.insert(src, -1, 0, axis=1)
	src = np.insert(src, -1, 0, axis=1)
	print(src.shape)

	dst = np.zeros(src.shape + (3,))

	for i in range(2, src.shape[0] - 2):
		for j in range(2, src.shape[1] - 2):
			if np.mod(i + j, 2) == 0:  # R or B
				alp = abs(-src[i - 2, j] + 2 * src[i, j] - src[i + 2, j]) + abs(src[i - 1, j] - src[i + 1, j])
				bet = abs(-src[i, j - 2] + 2 * src[i, j] - src[i, j + 2]) + abs(src[i, j - 1] - src[i, j + 1])
				if alp < bet:
					gout = (src[i - 1, j] + src[i + 1, j]) / 2 + (-src[i - 2, j] + 2 * src[i, j] - src[i + 2, j]) / 4
				elif alp > bet:
					gout = (src[i, j - 1] + src[i, j + 1]) / 2 + (-src[i, j - 2] + 2 * src[i, j] - src[i, j + 2]) / 4
				else:
					gout = (src[i - 1, j] + src[i + 1, j] + src[i, j - 1] + src[i, j + 1]) / 4 + (
							-src[i - 2, j] - src[i, j - 2] + 4 * src[i, j] - src[i + 2, j] - src[i, j + 2]) / 8

				dst[i, j, 1] = gout
			else:  # G

				dst[i, j, 1] = src[i, j]
	print('G')
	print(dst.max(), dst.min())
	print(src.max(), src.min())
	n = 0
	for i in range(2, src.shape[0] - 2):
		for j in range(2, src.shape[1] - 2):
			if np.mod(i, 2) == 0:
				if np.mod(j, 2) == 0:
					pos = (0, 0)  # 'R'
					dst[i, j, 0] = src[i, j]
					alp = abs(-dst[i - 1, j + 1, 1] + 2 * dst[i, j, 1] - dst[i + 1, j - 1, 1]) + abs(
						src[i - 1, j + 1] - src[i + 1, j - 1])
					bet = abs(-dst[i - 1, j - 1, 1] + 2 * dst[i, j, 1] - dst[i + 1, j + 1, 1]) + abs(
						src[i - 1, j - 1] - src[i + 1, j + 1])
					if alp < bet:
						bout = (src[i - 1, j + 1] + src[i + 1, j - 1]) / 2 + (
								-dst[i - 1, j + 1, 1] + 2 * dst[i, j, 1] - dst[i + 1, j - 1, 1]) / 4
					elif alp > bet:
						bout = (src[i - 1, j - 1] + src[i + 1, j + 1]) / 2 + (
								-dst[i - 1, j - 1, 1] + 2 * dst[i, j, 1] - dst[i + 1, j + 1, 1]) / 4
					else:
						bout = (src[i - 1, j + 1] + src[i + 1, j - 1] + src[i - 1, j - 1] + src[i + 1, j + 1]) / 4 + (
								-dst[i - 1, j - 1, 1] - dst[i - 1, j + 1, 1] + 4 * dst[i, j, 1] - dst[
							i + 1, j - 1, 1] - dst[i + 1, j + 1, 1]) / 8
					dst[i, j, 2] = bout

				else:
					pos = (0, 1)  # G
					dst[i, j, 0] = (src[i, j - 1] + src[i, j + 1]) / 2 + (
							-dst[i, j - 1, 1] + 2 * dst[i, j, 1] - dst[i, j + 1, 1]) / 4
					dst[i, j, 2] = (src[i - 1, j] + src[i + 1, j]) / 2 + (
							-dst[i - 1, j, 1] + 2 * dst[i, j, 1] - dst[i + 1, j, 1]) / 4
			else:
				if np.mod(j, 2) == 0:
					pos = (1, 0)  # G
					dst[i, j, 0] = (src[i - 1, j] + src[i + 1, j]) / 2 + (
							-dst[i - 1, j, 1] + 2 * dst[i, j, 1] - dst[i + 1, j, 1]) / 4
					dst[i, j, 2] = (src[i, j - 1] + src[i, j + 1]) / 2 + (
							-dst[i, j - 1, 1] + 2 * dst[i, j, 1] - dst[i, j + 1, 1]) / 4
				else:
					pos = (1, 1)  # B
					alp = abs(-dst[i - 1, j + 1, 1] + 2 * dst[i, j, 1] - dst[i + 1, j - 1, 1]) + abs(
						src[i - 1, j + 1] - src[i + 1, j - 1])
					bet = abs(-dst[i - 1, j - 1, 1] + 2 * dst[i, j, 1] - dst[i + 1, j + 1, 1]) + abs(
						src[i - 1, j - 1] - src[i + 1, j + 1])
					if alp < bet:
						rout = (src[i - 1, j + 1] + src[i + 1, j - 1]) / 2 + (
								-dst[i - 1, j + 1, 1] + 2 * dst[i, j, 1] - dst[i + 1, j - 1, 1]) / 4
					elif alp > bet:
						rout = (src[i - 1, j - 1] + src[i + 1, j + 1]) / 2 + (
								-dst[i - 1, j - 1, 1] + 2 * dst[i, j, 1] - dst[i + 1, j + 1, 1]) / 4
					else:
						rout = (src[i - 1, j + 1] + src[i + 1, j - 1] + src[i - 1, j - 1] + src[i + 1, j + 1]) / 4 + (
								-dst[i - 1, j - 1, 1] - dst[i - 1, j + 1, 1] + 4 * dst[i, j, 1] - dst[
							i + 1, j - 1, 1] - dst[i + 1, j + 1, 1]) / 8
					dst[i, j, 0] = rout
					dst[i, j, 2] = src[i, j]
	print('n={}'.format(n))
	print(dst.max(), dst.min())
	# dst = (dst - dst.min()) * 255 / (dst.max()-dst.min())
	dst = np.delete(dst, 0, axis=0)
	dst = np.delete(dst, 0, axis=0)
	dst = np.delete(dst, 0, axis=0)
	dst = np.delete(dst, -1, axis=0)
	dst = np.delete(dst, -1, axis=0)
	dst = np.delete(dst, -1, axis=0)
	dst = np.delete(dst, 0, axis=1)
	dst = np.delete(dst, 0, axis=1)
	dst = np.delete(dst, 0, axis=1)
	dst = np.delete(dst, -1, axis=1)
	dst = np.delete(dst, -1, axis=1)
	dst = np.delete(dst, -1, axis=1)
	print(dst.shape)

	return dst


def xtr_func(src):
	dst = np.zeros(src.shape)
	for (i, ch) in enumerate(['R', 'G', 'B']):
		if ch == 'G':
			karnel = np.array([[0.25, 0.5, 0.25],
			                   [0.5, 1, 0.5],
			                   [0.25, 0.5, 0.25]])
		elif ch == 'B':
			karnel = np.array([[0, 0.25, 0],
			                   [0.25, 1, 0.25],
			                   [0, 0.25, 0]])
		else:
			karnel = np.array([[0.25, 0.5, 0.25],
			                   [0.5, 1, 0.5],
			                   [0.25, 0.5, 0.25]])
		dst[:, :, i] = cv2.filter2D(src[:, :, i], -1, karnel)
	return dst


def rgb2bayer(img, rgb=0):
	bayer_rgb = np.zeros(img.shape)

	for (i, ch) in enumerate(['R', 'G', 'B']):
		if ch == 'R':
			pat = np.array([[1, 0],
			                [0, 0]])
		elif ch == 'G':
			pat = np.array([[0, 1],
			                [1, 0]])
		else:
			pat = np.array([[0, 0],
			                [0, 1]])
		mesh = mesh_maker(img.shape[:2], pat)
		bayer_rgb[:, :, i] = img[:, :, i] * mesh

	if rgb:
		bayer = bayer_rgb
	else:
		bayer = bayer_rgb[:, :, 0] + bayer_rgb[:, :, 1] + bayer_rgb[:, :, 2]

	return bayer


def rgb2xtrans(img, rgb=0):
	xtrans_rgb = np.zeros(img.shape)

	for (i, ch) in enumerate(['R', 'G', 'B']):
		if ch == 'R':
			pat = np.array([[0, 0, 0, 0, 1, 0],
			                [1, 0, 1, 0, 0, 0],
			                [0, 0, 0, 0, 1, 0],
			                [0, 1, 0, 0, 0, 0],
			                [0, 0, 0, 1, 0, 1],
			                [0, 1, 0, 0, 0, 0]])
		elif ch == 'G':
			pat = np.array([[1, 0, 1],
			                [0, 1, 0],
			                [1, 0, 1]])
		else:
			pat = np.array([[0, 1, 0, 0, 0, 0],
			                [0, 0, 0, 1, 0, 1],
			                [0, 1, 0, 0, 0, 0],
			                [0, 0, 0, 0, 1, 0],
			                [1, 0, 1, 0, 0, 0],
			                [0, 0, 0, 0, 1, 0]])
		mesh = mesh_maker(img.shape[:2], pat)
		xtrans_rgb[:, :, i] = img[:, :, i] * mesh

	if rgb:
		xtrans = xtrans_rgb
	else:
		xtrans = xtrans_rgb[:, :, 0] + xtrans_rgb[:, :, 1] + xtrans_rgb[:, :, 2]

	return xtrans


def mesh_maker(shape, pattern):
	out = np.zeros(shape)
	for i in range(out.shape[0]):
		for j in range(out.shape[1]):
			out[i, j] = pattern[np.mod(i, pattern.shape[0]), np.mod(j, pattern.shape[1])]
	return out


def img_show2(img1, img2, titles=['SRC', 'DST']):
	fig = plt.figure(figsize=(20, 7))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)

	ax1.set_title(titles[0]), ax2.set_title(titles[1])

	ax1.set_xticks([]), ax1.set_yticks([])
	ax2.set_xticks([]), ax2.set_yticks([])

	ax1.imshow(img1)
	ax2.imshow(img2)

	plt.show()


def img_show3(img1, img2, img3, titles=['SRC', '', 'DST']):
	fig = plt.figure(figsize=(20, 7))
	ax1 = fig.add_subplot(131)
	ax2 = fig.add_subplot(132)
	ax3 = fig.add_subplot(133)

	ax1.set_title(titles[0]), ax2.set_title(titles[1]), ax3.set_title(titles[2])

	ax1.set_xticks([]), ax1.set_yticks([])
	ax2.set_xticks([]), ax2.set_yticks([])
	ax3.set_xticks([]), ax3.set_yticks([])

	ax1.imshow(img1)
	ax2.imshow(img2, cmap='gray')
	ax3.imshow(img3)

	plt.show()

def im_save(img, file):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.set_xticks([]), ax1.set_yticks([])
	print(img.max(),img.min(), img.shape)
	img = img.astype('uint8')
	ax1.imshow(img, cmap='gray')
	fig.savefig(file)

def im_save_waku(img, file, lss):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.set_xticks([]), ax1.set_yticks([])
	print(img.max(),img.min(), img.shape)
	img = img.astype('uint8')
	ax1.imshow(img, cmap='gray')

	for ls in lss:
		x0 = ls[2]
		y0 = ls[0]
		print(x0,y0)
		w = abs(ls[3] - ls[2])
		h = abs(ls[1] - ls[0])
		r = patches.Rectangle(xy=(x0, y0), width=w, height=h, ec='#FF0000', fill=False)
		ax1.add_patch(r)
	fig.savefig(file)

if __name__ == '__main__':
	waku()
