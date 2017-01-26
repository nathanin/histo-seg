#!/usr/bin/python

# /home/nathan/mzmo/data/source/make_data.py

import os
import glob
import cv2
import numpy as np
import colorNormalization as cn
import itertools

# M0 / M1 project:
# images @ 1k x 1k @  40X
# Close to 512 at 20X


writesize = 256

def run_for_segmentation(m0in, m1in):
	m0out = '/home/nathan/mzmo/data/seg/m0'
	m1out = '/home/nathan/mzmo/data/seg/m1'

	listm0 = os.path.join(m0out, 'list.txt')
	listm1 = os.path.join(m1out, 'list.txt')

	sources = [m0in, m1in]
	destinations = [m0out, m1out]
	listfiles = [listm0, listm1]

	for src, dst, lst in zip(sources, destinations, listfiles):
	# List out contents
		if not os.path.exists(dst):
			os.makedirs(dst)

		search = os.path.join(src, '*.tif')
		files = glob.glob(search)

	# Create the blank dummy file
		dummy = np.zeros(shape=(256,256))
		dummy_name = os.path.join(dst, 'dummy.png')
		cv2.imwrite(dummy_name, dummy)

	# Copy files over; open the list file
		with open(lst, 'w') as lf:
			for f in files:
				base = os.path.basename(f)
				base = base.replace(' ', '_')
				base = base.replace('.tif', '.jpg')
				img = cv2.imread(f)
		# Color normalize
				img = cn.normalize(img)

		# Resize directly from 1k --> 256
				img = cv2.resize(img, dsize=(256, 256))
				img_name = os.path.join(dst, base)
				print img_name
				cv2.imwrite(img_name, img)
				lf.write('{} {}\n'.format(img_name, dummy_name))



def run_for_AlexNet(m0in, m1in, paths, p):
	'''
	Partition images into three catagories according to set percents
	'''
	print "Running partitioning code for AlexNet training & validation"
	#training = '/home/nathan/mzmo/data/training'
	#testing = '/home/nathan/mzmo/data/testing'
	#validation = '/home/nathan/mzmo/data/validation'

#	paths = [training, testing, validation]

	M0_FILES = glob.glob(os.path.join(m0in, '*.tif'))
	M1_FILES = glob.glob(os.path.join(m1in, '*.tif'))

# There will be an un-even number of tiles in each sub-division this way,
# if M1 and M0 are uneven. Which is the case. Boo.!	
	r = range(len(paths))	
	M0_PARTITION = np.random.choice(r, size=(len(M0_FILES),), p=p)
	M1_PARTITION = np.random.choice(r, size=(len(M1_FILES),), p=p)

	for index, path in enumerate(paths):
		M0_SUB = itertools.compress(M0_FILES, M0_PARTITION==index)
		M1_SUB = itertools.compress(M1_FILES, M1_PARTITION==index)
		listfile = os.path.join(path, 'list.txt')
		# index = 1
		with open(listfile, 'w') as lf:
			for f in M0_SUB:
				st = os.path.basename(f)
				st = st.replace(' ', '_')
				st = st.replace('(', '__')
				st = st.replace(')', '__')
				st = st.replace('.tif', '.jpg')

				print "M0: {} to {}".format(st, path)
				lf.write('{} 0\n'.format(st))

				st = os.path.join(path, st)

				img = cv2.imread(f)
				# img = cn.normalize(img)
				img = cv2.resize(img, dsize=(writesize, writesize))
				cv2.imwrite(st, img)


			for f in M1_SUB:
				st = os.path.basename(f)
				st = st.replace(' ', '_')
				st = st.replace('(', '__')
				st = st.replace(')', '__')
				st = st.replace('.tif', '.jpg')

				print "M1: {} to {}".format(st, path)
				lf.write('{} 1\n'.format(st))

				st = os.path.join(path, st)
				
				img = cv2.imread(f)
				img = cv2.resize(img, dsize=(writesize, writesize))
				# img = cn.normalize(img)
				cv2.imwrite(st, img)

def data_flip(t):
	pass


def data_rotate(t):
	center = (writesize/2, writesize/2)
	rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=90, scale=1.0)

	img_list = glob.glob(os.path.join(t, '*.jpg'))
	for name in img_list:
		img = cv2.imread(name)
		#print 'Rotating {}'.format(name)
		for k in range(3):
			name = name.replace('.jpg', 'r.jpg')
#			print name
			img = cv2.warpAffine(src=img, M=rotation_matrix, dsize=(img.shape[0:2]))
			cv2.imwrite(filename=name, img=img)	
	print 'Done rotating'



def data_coloration(t):
	l_mean_range = (144.048, 130.22)
	l_std_range = (40.23, 35.00)

	img_list = glob.glob(os.path.join(t, '*.jpg'))
	for name in img_list:
		img = cv2.imread(name)
		print 'Coloration {}'.format(name)
		for LMN, LSTD in zip(l_mean_range, l_std_range):
			name = name.replace('.jpg', 'c.jpg')
#			print name
			target = np.array([ [LMN, LSTD], [169.3, 9.01], [105.97, 6.67] ])
			img = cn.normalize(img, target)
			cv2.imwrite(filename=name, img=img)
	print 'Done colorizing'


def write_listfile(t):
	'''
	listfile will already exist; We need to maintain the base-name + class relationship
	
	1. Open list.txt
	2. Create new list2.txt
	3. Scan list.txt and find the number of matches with [base-name]* (regex? or just glob)
	4. Write into list2.txt the contents of the glob, along with the correct classes. 
	5. Delete list.txt
	6. Rename list2.txt to list.txt
	
	'''
	listfile = os.path.join(t, 'list.txt')	
	listfile2 = os.path.join(t, 'list2.txt')
	print "Transferring classes from {} to {}".format(listfile, listfile2)	
	f2 = open(listfile2, 'w')	


	with open(listfile, 'r') as f1:
		for line in f1:
			base, label = line.split(' ')
			base = base.replace('.jpg', '')
			#print base
			match = os.path.join(t, '{}*'.format(base))
			#print match
			match = glob.glob(match)
			#counter=1	
			for m in match:
				#print m
				base = os.path.basename(m)
				st = '{} {}'.format(base, label.replace(' ',''))
				#print '{} {} {}'.format(counter, len(match), st)
				f2.write(st)
				#counter+=1	
	f2.close()	
	os.rename(listfile2, listfile)

def check_listfile(t):
	'''
	Some weirdness is happnening in `write_listfile` -
	Scan it and remove lines that aren't contained in the folder ~~

	'''
	listfile = os.path.join(t, 'list.txt')
#	listfile2 = os.path.join(t, 'list2.txt')
	
	print "Checking listfile {}".format(listfile)
	f1 = open(listfile, 'r')
#	f2 = open(listfile2, 'w')
	counter=0	
	for line in f1:
		match, label = line.split(' ')
		path = os.path.join(t, match)
		#print path
		if not os.path.exists(path):
			counter+=1
			print 'Error on {}'.format(line)
	print "Found {} errant entries".format(counter)
	f1.close()
#	f2.close()

def find_min(lst):
	low = lst[0]
	loc = 0
	for index, i in enumerate(lst):
		if i < low:
			low = i
			loc = index
	return loc 


def trim_classes(t):
	'''
	Given a directory with images, and a list.txt:
	- Sample an equal number of each class from list.txt;
	- Delete extra images
	- Write out a new list.txt

	'''
	listfile = os.path.join(t, 'list.txt')
	balanced_list = os.path.join(t, 'list2.txt')
	
	print "Balancing classes in {}".format(t)
	print "Building class lists:"	
	img_lists = [ [], [] ]
	with open(listfile, 'r') as f:
		for l in f:
			img, label = l.split(' ')
			img_lists[int(label)].append(img)

	# Get the lengths of each list:
	members = [len(l) for l in img_lists]
	low = find_min(members)
	if low==0: # TODO Magic numbers
		lrg = 1
	else:
		lrg = 0
	print 'Class {} has only {} members'.format(low, members[low])
	print 'Class {} has {} members.'.format(lrg, members[lrg]) 

	# Write out the shorter list:
	with open(balanced_list, 'w') as f:	
		for n in img_lists[low]:
			f.write('{} {}\n'.format(n, low)) 

	# Delete some from the shorter list `1` means we keep it:
	ratio = members[low]/float(members[lrg]) #-1 is safe... ?
	p = [1-ratio, ratio]
	PARTITION = np.random.choice([0,1], size=(members[lrg],), p=p)
	print 'partitioning {} of {} files'.format(PARTITION.sum(), PARTITION.size)	
	with open(balanced_list, 'a') as f:
		for choice, img in zip(PARTITION, img_lists[lrg]):
			#print '{}\t{}'.format(choice, img)
			if choice == 1:
				f.write('{} {}\n'.format(img, lrg))
			elif os.path.exists(os.path.join(t, img)):
				#print 'Removing {}'.format(img)
				os.remove(os.path.join(t, img))	

	# Replace the new and old list files. 
	os.rename(balanced_list, listfile)	

	
def multiply_data(targets):
	'''
	Multiply the images in a folder with some combination of transforms.
	'''
	for t in targets:
		data_rotate(t)
		data_coloration(t)
		write_listfile(t)
		#check_listfile(t)
		trim_classes(t)	


####

#
m0 = '/home/nathan/mzmo/data/source/m0'
m1 = '/home/nathan/mzmo/data/source/m1'

#training = '/home/nathan/mzmo/data/training'
#validation = '/home/nathan/mzmo/data/validation'
#paths = [training, validation]
#p = [0.7,0.3]
#run_for_AlexNet(m0, m1, paths, p)

#multiply_data(paths)

run_for_segmentation(m0, m1)


