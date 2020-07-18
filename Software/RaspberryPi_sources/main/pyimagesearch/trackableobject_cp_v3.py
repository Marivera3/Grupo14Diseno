class TrackableObject:
	def __init__(self, objectID, centroid, rc, imagen, proba, dev):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		self.pic = imagen
		self.prob = proba
		self.name = rc
		self.block_n = False
		self.device = dev
		# initialize a boolean used to indicate if the object has
		# already been counted or not
		self.counted = False
		self.sent = False
		self.out = False
		self.inn = False
		if rc != 'unknown':
			self.reconocido = True
		else:
			self.reconocido = False
