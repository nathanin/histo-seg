# /home/nathan/_misc_python/colorNormalization.py
# /Users/nathaning/local_projects/semantic-pca/code/colorNormalization.py


from scipy import misc
# from skimage import color
import skimage as ski
import numpy as np
import cv2

def normalize(image,target=None,verbose=False):
	if target is None:
		target = np.array([[148.60, 41.56], [169.30, 9.01],[105.97, 6.67]])
		# target = np.array( [[68.33, 23.10], [19.09, 16.44], [-11.33, 8.97]] )
   	if verbose:
   		print "Color normalization:"
   		# print "Recieved {} image".format(image.dtype),
   		print "Recieved {} image".format(type(image)),

	M = image.shape[0]
	N = image.shape[1]

	if verbose:
		print " size {}rows, {}cols".format(str(M), str(N))

	whitemask = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	whitemask = whitemask > 215
   
	if verbose:
		print "Taking it into LAB space"
	imagelab = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
   
	imageL,imageA,imageB = cv2.split(imagelab)
   
	imageLM = np.ma.MaskedArray(imageL,whitemask) # mask is valid when true
	imageAM = np.ma.MaskedArray(imageA,whitemask)
	imageBM = np.ma.MaskedArray(imageB,whitemask)
	
	imageLMean  = imageLM.mean()
	imageLSTD   = imageLM.std()
   
	imageAMean  = imageAM.mean()
	imageASTD   = imageAM.std()
   
	imageBMean  = imageBM.mean()
	imageBSTD   = imageBM.std()

	if verbose:
		print "Lmean: {}, Lstd: {}".format(str(imageLMean), str(imageLSTD))
		print "Amean: {}, Astd: {}".format(str(imageAMean), str(imageASTD))
		print "Bmean: {}, Bstd: {}".format(str(imageBMean), str(imageBSTD))

	# normalization in lab
	imageL = (imageL-imageLMean)/imageLSTD*target[0][1]+target[0][0]
	imageA = (imageA-imageAMean)/imageASTD*target[1][1]+target[1][0]
	imageB = (imageB-imageBMean)/imageBSTD*target[2][1]+target[2][0]
   

	if verbose:
		print "Image adjusted, going back to RGB...",

	imagelab = cv2.merge((imageL,imageA,imageB))
	imagelab = np.clip(imagelab, 0, 255)
	imagelab = imagelab.astype(np.uint8)

	# imagelab = np.zeros((M,N,3), dtype=np.float64)
	# imagelab = np.array((imageL, imageA, imageB))
	# imagelab[:,:,0] = imageL
	# imagelab[:,:,1] = imageA
	# imagelab[:,:,2] = imageB

	returnimage = cv2.cvtColor(imagelab,cv2.COLOR_LAB2RGB)
	# returnimage = color.lab2rgb(imagelab)
	returnimage[whitemask]=image[whitemask]
   
	if verbose:
		print "Done."
	return returnimage
