"""Mask reconciliation and measurement on the CAE station-to-UTM mapping."""
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import csv

import cv2
import numpy as np


@dataclass
class Patch:
    x: int
    y: int
    mask: np.ndarray
    kind: str
    score: float
    sources: list
    flags: set = field(default_factory=set)

    @property
    def bounds(self):
        h, w = self.mask.shape
        return (self.x, self.y, self.x+w, self.y+h)


def trimmed(x, y, mask, kind, score, sources, flags=None):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    a, b, c, d = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
    return Patch(x+a, y+b, mask[b:d,a:c].copy(), kind, score, sources, set(flags or ()))


def intersect(a, b):
    x0,y0,x1,y1 = a.bounds
    X0,Y0,X1,Y1 = b.bounds
    x, y, r, d = max(x0,X0), max(y0,Y0), min(x1,X1), min(y1,Y1)
    if x >= r or y >= d:
        return None
    return (a.mask[y-y0:d-y0,x-x0:r-x0], b.mask[y-Y0:d-Y0,x-X0:r-X0])


def seam_pixels(p, axis, border, before, band):
    # Return (cross-seam coordinate, along-seam coordinate) pairs near the cut.
    mask = p.mask if axis == 0 else p.mask.T
    offset = p.x if axis == 0 else p.y
    other = p.y if axis == 0 else p.x
    lo, hi = (border-band,border) if before else (border,border+band)
    a, b = max(0,lo-offset), min(mask.shape[1],hi-offset)
    if b <= a:
        return np.empty((0,2), dtype=int)
    y, x = np.nonzero(mask[:,a:b])
    return np.column_stack((x+a+offset,y+other))


def seam_match(a, b, axis, border, gap):
    p = seam_pixels(a,axis,border,True,gap+1)
    q = seam_pixels(b,axis,border,False,gap+1)
    if not len(p) or not len(q):
        return False
    if int(q[:,0].min()-p[:,0].max()-1) > gap:
        return False
    if a.kind != "linear_crack":
        # Area masks must share support along the cut. Parallel nearby masks are insufficient.
        common = np.intersect1d(p[:,1],q[:,1])
        return len(common) >= (2 if a.kind == "pothole" else 4)
    p0 = np.median(p[p[:,0]==p[:,0].max(),1])
    q0 = np.median(q[q[:,0]==q[:,0].min(),1])
    if abs(p0-q0) > gap+1:
        return False
    # Reject conflicting directions when both short trajectories provide a stable fit.
    slopes = []
    for patch, before in ((a,True),(b,False)):
        points = seam_pixels(patch,axis,border,before,24)
        cross = np.unique(points[:,0])
        if len(cross) >= 8:
            along = [np.median(points[points[:,0]==x,1]) for x in cross]
            slopes.append(np.polyfit(cross-cross[0],along,1)[0])
    return len(slopes) < 2 or abs(np.arctan(slopes[0])-np.arctan(slopes[1])) <= np.pi/4


def reconcile(patches, grid, gap):
    """Join unambiguous same-class neighbors, then enforce within-class pixel ownership."""
    parent = list(range(len(patches)))
    by_tile = defaultdict(list)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i,j):
        a,b = root(i),root(j)
        if a != b:
            parent[max(a,b)] = min(a,b)

    for i,p in enumerate(patches):
        by_tile[p.sources[0]["tile_origin"]].append(i)
    for (x,y), (w,h) in grid.items():
        ids = by_tile.get((x,y), ())
        for pos,i in enumerate(ids):
            for j in ids[pos+1:]:
                a,b = patches[i], patches[j]
                if a.kind == b.kind:
                    pair = intersect(a,b)
                    if pair and np.count_nonzero(pair[0]&pair[1]) >= .5*min(a.mask.sum(),b.mask.sum()):
                        join(i,j)
        for axis, neighbor, border in ((0,(x+w,y),x+w),(1,(x,y+h),y+h)):
            if neighbor not in grid:
                continue
            other_ids = by_tile.get(neighbor, ())
            pairs = set()
            for i in ids:
                for j in other_ids:
                    a,b = patches[i],patches[j]
                    if a.kind == b.kind and seam_match(a,b,axis,border,gap):
                        pairs.add((root(i),root(j)))
            left_count = Counter(i for i,j in pairs)
            right_count = Counter(j for i,j in pairs)
            for members, counter, before in ((ids,left_count,True),(other_ids,right_count,False)):
                for i in members:
                    if not counter[root(i)] and len(seam_pixels(patches[i],axis,border,before,gap+1)):
                        patches[i].flags.add("unmatched_tile_edge")
            for i,j in sorted(pairs):
                if left_count[i] == right_count[j] == 1:
                    join(i,j)
                else:
                    patches[i].flags.add("ambiguous_tile_seam")
                    patches[j].flags.add("ambiguous_tile_seam")
    groups = defaultdict(list)
    for i,p in enumerate(patches):
        groups[root(i)].append(p)
    merged = []
    for group in groups.values():
        # A single instance keeps its exact original mask, including holes.
        x,y,r,b = zip(*(p.bounds for p in group))
        x0,y0 = min(x),min(y)
        canvas = np.zeros((max(b)-y0,max(r)-x0),dtype=bool)
        for p in group:
            a,c,d,e = p.bounds
            canvas[c-y0:e-y0,a-x0:d-x0] |= p.mask
        merged.append(Patch(x0,y0,canvas,group[0].kind,max(p.score for p in group),
                            [s for p in group for s in p.sources],set().union(*(p.flags for p in group))))
    kept = []
    for p in sorted(merged,key=lambda p: (-p.score,p.y,p.x)):
        for prior in kept:
            pair = intersect(p,prior)
            if not pair or not np.any(pair[0]&pair[1]):
                continue
            if p.kind == prior.kind:
                pair[0][:] &= ~pair[1]
                p.flags.add("same_class_overlap_resolved_by_score")
                prior.flags.add("same_class_overlap_resolved_by_score")
            else:
                p.flags.add("cross_class_overlap_review")
                prior.flags.add("cross_class_overlap_review")
        p = trimmed(p.x,p.y,p.mask,p.kind,p.score,p.sources,p.flags)
        if p is not None:
            kept.append(p)
    return sorted(kept,key=lambda p: (p.x,p.y,p.kind,-p.score))


class Mapping:
    def __init__(self, path, transform, width, height):
        with path.open(newline="",encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        columns = ("station_m","center_easting_m","center_northing_m",
                   "left_unit_easting","left_unit_northing")
        self.s, self.e, self.n, self.ne, self.nn = np.array(
            [[float(row[k]) for k in columns] for row in rows]).T
        if not np.isfinite([self.s,self.e,self.n,self.ne,self.nn]).all() or np.any(np.diff(self.s)<=0):
            raise ValueError("Invalid station mapping")
        self.t = transform
        if (transform.a <= 0 or transform.e >= 0 or transform.b or transform.d
                or transform.c < self.s[0]-1e-8 or transform.c+width*transform.a > self.s[-1]+1e-8
                or not transform.f >= 0 >= transform.f+height*transform.e):
            raise ValueError("TIFF must be the CAE unfolded raster on the supplied station grid")

    def xy(self, x, y):
        station = self.t.c+np.asarray(x)*self.t.a
        offset = self.t.f+np.asarray(y)*self.t.e
        return (np.interp(station,self.s,self.e)+offset*np.interp(station,self.s,self.ne),
                np.interp(station,self.s,self.n)+offset*np.interp(station,self.s,self.nn))

    def area(self, patch):
        area, folds = 0., 0
        for start in range(0,patch.mask.shape[0],128):
            row,col = np.nonzero(patch.mask[start:start+128])
            x,y = col+patch.x,row+patch.y+start
            e0,n0 = self.xy(x,y)
            e1,n1 = self.xy(x+1,y)
            e2,n2 = self.xy(x+1,y+1)
            e3,n3 = self.xy(x,y+1)
            a = -.5*((e1-e0)*(n2-n0)-(n1-n0)*(e2-e0)+(e2-e0)*(n3-n0)-(n2-n0)*(e3-e0))
            area += float(np.abs(a).sum())
            folds += int(np.count_nonzero(a <= 0))
        return area,folds


def thin(mask):
    image = np.pad(mask.astype(np.uint8),1)
    # Use the same Zhang-Suen lookup implementation on all machines.
    # The reference measurements were made without opencv-contrib.
    # Zhang-Suen thinning with lookup tables, for OpenCV builds without ximgproc.
    codes = np.arange(256)[:,None]
    p = ((codes >> np.arange(8)) & 1).astype(bool)
    valid = ((p.sum(axis=1)>=2)&(p.sum(axis=1)<=6)
             & ((~p & np.roll(p,-1,axis=1)).sum(axis=1)==1))
    rules = (valid & ~(p[:,0]&p[:,2]&p[:,4]) & ~(p[:,2]&p[:,4]&p[:,6]),
             valid & ~(p[:,0]&p[:,2]&p[:,6]) & ~(p[:,0]&p[:,4]&p[:,6]))
    kernel = np.array([[128,1,2],[64,0,4],[32,16,8]],np.float32)
    while True:
        changed = False
        for lut in rules:
            code = cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_CONSTANT)
            remove = (image>0) & lut[code]
            if remove.any():
                image[remove] = 0
                changed = True
        if not changed:
            return image[1:-1,1:-1]>0


def centerline_metrics(patch, mapper):
    line = thin(patch.mask)
    padded = np.pad(line,1)
    r,c = np.nonzero(line)
    x,y = c+patch.x+.5,r+patch.y+.5
    e,n = mapper.xy(x,y)
    total = 0.
    for dx,dy in ((1,0),(0,1),(1,1),(-1,1)):
        connected = padded[r+1+dy,c+1+dx]
        if dx and dy:
            connected &= ~(padded[r+1,c+1+dx] | padded[r+1+dy,c+1])
        e2,n2 = mapper.xy(x[connected]+dx,y[connected]+dy)
        total += float(np.hypot(e2-e[connected],n2-n[connected]).sum())
    components = cv2.connectedComponents(line.astype(np.uint8), connectivity=8)[0]-1
    return total,components


def max_span(patch, mapper):
    mask = patch.mask.astype(np.uint8)
    points = cv2.findNonZero(mask)
    hull = cv2.convexHull(points).reshape(-1,2)
    e,n = mapper.xy(hull[:,0]+patch.x+.5,hull[:,1]+patch.y+.5)
    # Maximum span of the mapped hull, used as context rather than ASTM average diameter.
    pts = np.column_stack((e-e[0],n-n[0])).astype(np.float32)
    hull = cv2.convexHull(pts).reshape(-1,2)
    longest = 0.
    for i in range(0,len(hull),128):
        diff = hull[i:i+128,None,:]-hull
        longest = max(longest,float(np.max(np.sum(diff**2,axis=-1))))
    return longest**.5
