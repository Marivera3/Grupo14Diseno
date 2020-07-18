from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
	def __init__(self, maxDisappeared=50, maxDistance=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.names = OrderedDict()
		self.name_counter = OrderedDict()
		self.block_name = OrderedDict()
		self.pictures = OrderedDict()
		self.probs = OrderedDict()
		self.devices = OrderedDict()

		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

		# store the maximum distance between centroids to associate
		# an object -- if the distance is larger than this maximum
		# distance we'll start to mark the object as "disappeared"
		self.maxDistance = maxDistance

	def register(self, centroid, nombre, image, pb, div):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.names[self.nextObjectID] = nombre
		self.name_counter[self.nextObjectID] = 1
		self.block_name[self.nextObjectID] = False
		self.pictures[self.nextObjectID] = image
		self.probs[self.nextObjectID] = pb
		self.devices[self.nextObjectID] = div
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.names[objectID]
		del self.name_counter[objectID]
		del self.block_name[objectID]
		del self.pictures[objectID]
		del self.probs[objectID]
		del self.devices[objectID]
		del self.disappeared[objectID]

	def update(self, rects, nombres, pics, probs, device):
		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			return self.objects, self.names, self.pictures, self.probs, self.devices, self.block_name

		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i], nombres[i], pics[i], probs[i], device[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			#print('Objects centroids ',oC)
			# compute the distance between each pair of object centroids
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]
			usedRows = set()
			usedCols = set()
			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue
				if D[row, col] > self.maxDistance:
					continue
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]

				if nombres[col] == self.names[objectID] and not self.block_name[objectID]:
					self.name_counter[objectID] += 1
					self.block_name[objectID]  = True if self.name_counter[objectID] == 2 else False
				else:
					self.name_counter[objectID]  = 0
					self.names[objectID] = nombres[col]

				self.disappeared[objectID] = 0
				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			else:
				for col in unusedCols:
					self.register(inputCentroids[col], nombres[col], pics[col],
					probs[col], device[col])

		# return the set of trackable objects
		return self.objects, self.names, self.pictures, self.probs, self.devices, self.block_name
