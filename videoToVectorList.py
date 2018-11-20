from imageToVector import Extractor


class Process():
    def __init__(self):
        self.x = Extractor()

    def convToVector(self, cap):
        ##Convert every frame in the video to a vector, return the list of vectors
        counter = 0
        check = True
        frame_list = []
        vectorlist = []
        while check:
            check, vid = cap.read()
            frame_list.append(vid)
            counter += 1

        g = 0
        for i in range(counter - 1):
            if g == 87: break
            vectorlist.append(self.x.extract(frame_list[g]))
            g += 1

        return vectorlist
