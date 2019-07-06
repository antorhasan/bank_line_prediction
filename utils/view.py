
class viz():
    '''change bgr to rgb of a 3 band image
        input : np.ndarray
        output : np.ndarray'''
    def __init__(self, raster):
        self.raster = raster
    
    def torgb(self):
        orig = self.raster
        temp = self.raster
        print(orig[:,:,0:1])
        orig[:,:,0:1] = temp[:,:,2:3]
        orig[:,:,1:2] = temp[:,:,1:2]
        orig[:,:,2:3] = temp[:,:,2:3]
        print(orig[:,:,0:1])
        return orig
