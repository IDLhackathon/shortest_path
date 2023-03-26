import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import pdb
import math
from skimage.draw import line
import scipy
from scipy.spatial import Delaunay


## reads image 'opencv-logo.png' as grayscale
img_cv = cv2.imread('label.png', 0)
img_cv_col =  cv2.cvtColor(cv2.imread('label.png'),cv2.COLOR_BGR2RGB)
img_mask = cv2.imread('mask.png', 0)

## Position to avoid, hardcoded for the moment
dd_bad = [[660,300],[410,400],[380,260],[800,200],[400,370]]


## Compute gradient of areas
kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
edges_x = cv2.filter2D(img_cv,cv2.CV_8U,kernelx)
edges_y = cv2.filter2D(img_cv,cv2.CV_8U,kernely)

## Extract edtes
points = []
acc=0
for ii in range(0,img_cv.shape[0]) :
    for jj in range(0,img_cv.shape[1]) :
        acc+=1
        vv = edges_x[ii][jj]
        if vv != 0 :
            if acc%5==0 :
                points.append([jj,ii])

## Compute the 2D delaunay triangulation                
points = np.array(points);
tri = Delaunay(points)

def compute_gravity(ss) :
    return [(points[ss[0]][0]+points[ss[1]][0] + points[ss[2]][0])/3.0,
             (points[ss[0]][1]+points[ss[1]][1] + points[ss[2]][1])/3.0]
def pdist(c1,c2) :
    return math.sqrt(math.pow(c1[0] -c2[0],2) + math.pow(c1[1] -c2[1],2))

centers = []
centers_score = []
## Compute the gravity center and the impact of areas
for ss in tri.simplices :
    gcenter = compute_gravity(ss)
    centers.append(gcenter)
    score = 0
    for cc in dd_bad :
     if pdist(cc,gcenter) < 60 :
         score = score + 1
    centers_score.append(score)

## Fill the mask according to the areas impact
for ii in range(0,img_cv.shape[0]) :
    for jj in range(0,img_cv.shape[1]) :
        kk = int(tri.find_simplex([jj,ii]))
        if centers_score[kk] > 0 :
            img_mask[ii,jj] = 255

centers = np.array(centers)
dd_bad = np.array(dd_bad)

## Source & Destination
src = int(tri.find_simplex([100,130]))
trg = int(tri.find_simplex([1000,300]))

## Shortest path initialization
nbrs =  tri.neighbors
G = nx.DiGraph()
id = 0
for ss in tri.simplices :
    G.add_node(id)
    id +=1

def edist(c1,c2) :
    rr, cc = line(int(c1[0]), int(c1[1]),int(c2[0]), int(c2[1]))
    arr = img_mask[cc,rr]
    ww = np.count_nonzero(arr)*1000
    return (math.sqrt(math.pow(c1[0] -c2[0],2) + math.pow(c1[1] -c2[1],2)) + ww)
    
for nn in nbrs :
    if -1 in nn : continue
    s0 = tri.simplices[nn[0]]
    s1 = tri.simplices[nn[1]]
    s2 = tri.simplices[nn[2]]

    dd1 = edist(compute_gravity(s0),compute_gravity(s1))
    dd2 = edist(compute_gravity(s0),compute_gravity(s2))
    dd3 = edist(compute_gravity(s1),compute_gravity(s2))

    G.add_edge(nn[0],nn[1],weight=dd1)
    G.add_edge(nn[1],nn[0],weight=dd1)
    G.add_edge(nn[0],nn[2],weight=dd2)
    G.add_edge(nn[2],nn[0],weight=dd2)
    G.add_edge(nn[1],nn[2],weight=dd3)
    G.add_edge(nn[2],nn[1],weight=dd3)    

## Shortest path computation
sp = nx.shortest_path(G,source=src,target=trg, weight='weight')

## Display
plt.imshow(img_cv_col)
plt.triplot(points[:,0], points[:,1], tri.simplices, lw=0.25,color="black")
plt.scatter(centers[src,0], centers[src,1],s=250,c="red")
plt.scatter(centers[trg,0], centers[trg,1],s=200,c="red")

if len(dd_bad) > 0 :
    plt.plot(dd_bad[:,0],dd_bad[:,1],"x", ms=16,mew=4,c="black")
for ii in range(0,len(sp)-1) :
    cc1 = centers[sp[ii]]
    cc2 = centers[sp[ii+1]]
    plt.plot([cc1[0],cc2[0]],[cc1[1],cc2[1]],linewidth=2,color="red")

plt.show()
