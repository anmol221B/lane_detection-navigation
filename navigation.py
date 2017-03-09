
# coding: utf-8

# In[118]:

import networkx as nx
import numpy as np
import pandas as pd
import json
import math
import smopy
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = mpl.rcParams['savefig.dpi'] = 300


# In[ ]:




# In[119]:

import gdal
import ogr
#g = nx.read_shp("/home/satyam/code/driverlessCar/laneDetection/navigation/roorkee-map-data/ex_CgDnF2Lgnv11eRRSj2q61gykLZYfx.imposm-shapefiles/ex_CgDnF2Lgnv11eRRSj2q61gykLZYfx_osm_roads.shp")


# In[120]:


sg=nx.Graph()
#g.add_edge((1,2),(2,3))
sg.add_edges_from([((77.895309,29.863463),(77.895351,29.863921)),((77.895351,29.863921),(77.895175,29.864159)),((77.895175,29.864159),(77.894163,29.864175)),((77.894163,29.864175),(77.893886,29.864180)),((77.893886,29.864180),(77.893431,29.864195)),((77.895309,29.863463),(77.895281,29.863103)),((77.895281,29.863103),(77.895246,29.862809)),((77.895246,29.862809),(77.895213,29.862369)),((77.895246,29.862809),(77.894177,29.862910)),((77.894177,29.862910),(77.894107,29.862512)),((77.894107,29.862512),(77.893676,29.862531)),((77.893676,29.862531),(77.893575,29.862444)),((77.893575,29.862444),(77.892477,29.862537)),((77.895213,29.862369),(77.895668,29.862100)),((77.895668,29.862100),(77.896768,29.862067)),((77.895668,29.862100),(77.895596,29.861186)),((77.895596,29.861186),(77.896733,29.861093)),((77.895596,29.861186),(77.895514,29.860648)),((77.896733,29.861093),(77.896696,29.860509)),((77.896733,29.861093),(77.896768,29.862067)),((77.896696,29.860509),(77.895515,29.860583)),((77.895514,29.860648),(77.895515,29.860583)),((77.896768,29.862067),(77.896790,29.862346)),((77.896790,29.862346),(77.896822,29.863044)),((77.896790,29.862346),(77.897854,29.862313)),((77.897854,29.862313),(77.897877,29.862985)),((77.896822,29.863044),(77.895281,29.863103)),((77.896822,29.863044),(77.897877,29.862985)),((77.897877,29.862985),(77.898457,29.862956)),((77.897877,29.862985),(77.897907,29.863466)),((77.897907,29.863466),(77.898489,29.863440)),((77.898489,29.863440),(77.898457,29.862956)),((77.897907,29.863466),(77.897943,29.864069)),((77.897943,29.864069),(77.896841,29.864086)),((77.896822,29.863044),(77.896841,29.864086)),((77.896841,29.864086),(77.895627,29.864125)),((77.895351,29.863921),(77.895627,29.864125)),((77.895627,29.864125),(77.895379,29.864327)),((77.895175,29.864159),(77.895379,29.864327)),((77.895213,29.862369),(77.894107,29.862512)),((77.893886,29.864180),(77.893676,29.862531))])

sg=sg.to_undirected()
print sg.edges()
#sg = list(nx.connected_component_subgraphs(g.to_undirected()))
#for i in range(0,len(sg)):
    #print sg[i].edges()
#len(sg)
#print sg[1].edges()


# In[121]:

pos0 = (29.864414, 77.895377)
#pos0 = (33.928887, -118.280360)
pos1 = (29.862428, 77.893463)
#pos1 = (34.0569, -118.2427)


# In[122]:

def get_path(n0, n1):
    """If n0 and n1 are connected nodes in the graph, this function
    return an array of point coordinates along the road linking
    these two nodes."""
    return np.array((sg[n0][n1]))


# In[129]:

EARTH_R = 6372.8
def geocalc(lat0, lon0, lat1, lon1):
    """Return the distance (in km) between two points in 
    geographical coordinates."""
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    dlon = lon0 - lon1
    y = np.sqrt(
        (np.cos(lat1) * np.sin(dlon)) ** 2
         + (np.cos(lat0) * np.sin(lat1) 
         - np.sin(lat0) * np.cos(lat1) * np.cos(dlon)) ** 2)
    x = np.sin(lat0) * np.sin(lat1) +         np.cos(lat0) * np.cos(lat1) * np.cos(dlon)
    c = np.arctan2(y, x)
    bearing = math.atan2(math.sin(lon1-lon0)*math.cos(lat1), math.cos(lat0)*math.sin(lat1)-math.sin(lat0)*math.cos(lat1)*math.cos(lon1-lon0)) 
    bearing = np.degrees(bearing) 
    bearing = (bearing + 360) % 360
    print bearing
    return EARTH_R * c


# In[130]:

def get_path_length(path):
    return np.sum(geocalc(path[1:,0], path[1:,1],
                          path[:-1,0], path[:-1,1]))


# In[131]:

# Compute the length of the road segments.
for n0, n1 in sg.edges_iter():
    #path = get_path(n0, n1)
    #distance = get_path_length(path)
    sg.edge[n0][n1]['distance'] = geocalc(n0[1], n0[0], n1[1], n1[0])
    print  sg.edge[n0][n1]['distance']
    


# In[111]:

nodes = np.array(sg.nodes())
# Get the closest nodes in the graph.
pos0_i = np.argmin(np.sum((nodes[:,::-1] - pos0)**2, axis=1))
pos1_i = np.argmin(np.sum((nodes[:,::-1] - pos1)**2, axis=1))


# In[112]:

# Compute the shortest path.
path = nx.shortest_path(sg, 
                        source=tuple(nodes[pos0_i]), 
                        target=tuple(nodes[pos1_i]),
                        weight='distance')
print (path)


# In[113]:

roads = pd.DataFrame([sg.edge[path[i]][path[i + 1]] 
                      for i in range(len(path) - 1)], 
                     columns=['FULLNAME', 'MTFCC', 
                              'RTTYP', 'distance'])
roads


# In[114]:

roads['distance'].sum()


# In[86]:

map = smopy.Map(pos0, pos1, z=7, margin=.1)


# In[14]:

def get_full_path(path):
    """Return the positions along a path."""
    p_list = []
    curp = None
    for i in range(len(path)-1):
        p = get_path(path[i], path[i+1])
        if curp is None:
            curp = p
        if np.sum((p[0]-curp)**2) > np.sum((p[-1]-curp)**2):
            p = p[::-1,:]
        p_list.append(p)
        curp = p[-1]
    return np.vstack(p_list)


# In[26]:

linepath = get_full_path(path)
x, y = map.to_pixels(linepath[:,1], linepath[:,0])


# In[231]:

plt.figure(figsize=(6,6));
map.show_mpl();
# Plot the itinerary.
plt.plot(x, y, '-k', lw=1.5);
# Mark our two positions.
plt.plot(x[0], y[0], 'ob', ms=10);
plt.plot(x[-1], y[-1], 'or', ms=10);


# In[ ]:



