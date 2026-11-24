"""Native RGB evidence and conservative restoration of previously reviewed lines."""
import cv2
import numpy as np
from rasterio.windows import Window
from .geometry import thin

CORE_DISTANCE_PX = 14
MIN_BROAD_CORE_PX = 1000
MIN_SEGMENT_SPAN_PX = 26
GAP = 6
PAD = 29
MIN_SPAN = 24

def luma_ridge(gray):
    response=np.zeros(gray.shape,np.float32)
    for sigma in (1.25,2.2,3.6,5.2):
        image=cv2.GaussianBlur(gray,(0,0),sigma)
        a=cv2.filter2D(image,-1,np.array([[1,-2,1]],np.float32))
        c=cv2.filter2D(image,-1,np.array([[1],[-2],[1]],np.float32))
        b=cv2.filter2D(image,-1,np.array([[1,0,-1],[0,0,0],[-1,0,1]],np.float32)*.25)
        d=np.sqrt((a-c)**2+4*b*b)
        across=(a+c+d)*.5
        along=(a+c-d)*.5
        positive=np.maximum(across,0)
        ratio=np.abs(along)/(positive+1e-6)
        strength=sigma*sigma*positive*np.exp(-1.8*ratio*ratio)
        np.maximum(response,strength,out=response)
    return response


def paint_border_band(rgb):
    im=rgb.astype(np.int16)
    r,g,b=(im[:,:,i] for i in range(3))
    warm=(r>155)&(g>145)&(r-b>45)&(g-b>35)
    band=(warm.mean(axis=1)>=.25).astype(np.uint8)[:,None]
    return cv2.morphologyEx(band,cv2.MORPH_GRADIENT,np.ones((25,1),np.uint8))>0


def line_kernel(length,c,s):
    center=(length-1)/2
    radius=(length-2)/2
    l=tuple(round(center-z*radius) for z in (c,s))
    r=tuple(round(center+z*radius) for z in (c,s))
    kernel=np.zeros((length,length),np.uint8)
    cv2.line(kernel,l,r,255,1,cv2.LINE_AA)
    return kernel.astype(np.float32)/float(kernel.sum())


def oriented_contrast(gray,paint_edge):
    image=cv2.GaussianBlur(gray,(0,0),.85)
    yy,xx=np.indices(gray.shape,dtype=np.float32)
    short=np.zeros(gray.shape,np.float32)
    long=np.zeros_like(short)
    for theta in np.arange(12)*np.pi/12:
        c,s=float(np.cos(theta)),float(np.sin(theta))
        for length,delta,output in ((17,5.,short),(27,7.5,short),(39,9.,long)):
            scan=cv2.filter2D(image,-1,line_kernel(length,c,s))
            left=cv2.remap(scan,xx+delta*s,yy-delta*c,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            right=cv2.remap(scan,xx-delta*s,yy+delta*c,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            strength=np.minimum(left,right)-scan
            if abs(s)<.3:
                strength=np.where(paint_edge,0,strength)
            np.maximum(output,strength,out=output)
    return short,long


def trace_visible(rgb,original):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    short,long=oriented_contrast(gray,paint_border_band(rgb))
    residual=cv2.GaussianBlur(gray,(0,0),8)-cv2.GaussianBlur(gray,(0,0),1)
    ridge=luma_ridge(gray)
    # Thresholds were selected by comparing source-image overlays for the
    # largest masks, faint asphalt texture and crops crossing yellow paint.
    weak=original&(short>=15.)&(long>=13.)&(residual>=5.5)&(ridge>=3.5)
    strong=weak&(short>=23.)&(long>=18.5)
    _,labels,stats,_=cv2.connectedComponentsWithStats(weak.astype(np.uint8),connectivity=8)
    keep=np.zeros(len(stats),bool)
    keep[np.unique(labels[strong])]=True
    keep &= (np.maximum(stats[:,2],stats[:,3])>=MIN_SEGMENT_SPAN_PX)&(stats[:,4]>=25)
    keep[0]=False
    # No closing or endpoint bridging: a connection may only follow adjacent
    # pixels that met the raw-image thresholds.
    return thin(keep[labels])


def source_view(src, x,y,r,b):
    return np.ascontiguousarray(src.read((1,2,3,4), window=Window(x,y,r-x,b-y)).transpose(1,2,0))


def local_frame(skeleton, y,x):
    signal = skeleton.astype(np.float32)
    ticks = np.arange(-8,9,dtype=np.float32)
    ones = np.ones_like(ticks)
    def moment(h,v):
        return cv2.sepFilter2D(signal,cv2.CV_32F,h,v,borderType=cv2.BORDER_CONSTANT)[y,x]
    n=moment(ones,ones)
    xx=moment(ticks,ones)/n
    yy=moment(ones,ticks)/n
    varx=moment(ticks*ticks,ones)/n-xx*xx
    vary=moment(ones,ticks*ticks)/n-yy*yy
    cov=moment(ticks,ticks)/n-xx*yy
    angle=.5*np.arctan2(2*cov,varx-vary)
    elongation=np.hypot(varx-vary,2*cov)/np.maximum(1.,varx+vary)
    return -np.sin(angle), np.cos(angle), (elongation>=.44)&(n>=6)


def supported_lines(rgb,valid,skeleton):
    y,x=np.nonzero(skeleton)
    if not len(x):
        return skeleton
    nx,ny,directional=local_frame(skeleton,y,x)
    raw=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray=cv2.GaussianBlur(raw,(0,0),.85)
    offsets=np.arange(-23,24)
    xx=np.clip(np.rint(x[:,None]+nx[:,None]*offsets).astype(np.int32),0,gray.shape[1]-1)
    yy=np.clip(np.rint(y[:,None]+ny[:,None]*offsets).astype(np.int32),0,gray.shape[0]-1)
    p=gray[yy,xx]
    v=valid[yy,xx]
    core=p[:,20:27]
    at_core=20+np.argmin(core,axis=1)
    darkest=core.min(axis=1)
    d=np.arange(len(x))
    allowed=directional & v[:,8:39].all(axis=1)
    near_l,near_r=np.median(p[:,12:17],axis=1),np.median(p[:,30:35],axis=1)
    far_l,far_r=np.median(p[:,4:11],axis=1),np.median(p[:,36:43],axis=1)
    close=np.minimum(near_l,near_r)-darkest
    far=np.minimum(far_l,far_r)-darkest
    # Both shoulders must be brighter than the dark opening. The intermediate
    # flanks constrain single-sided steps from paint, shadows and mosaicking.
    far_ok=(far>=10.)&(np.abs(far_l-far_r)<=np.maximum(14.,1.1*far))&v[:,4:43].all(axis=1)
    near_ok=(close>=8.)&(np.abs(near_l-near_r)<=np.maximum(13.,1.0*close))
    inner=p[d,at_core-2]+p[d,at_core+2]-2*darkest
    contrast=np.maximum(np.where(near_ok,close,0.), np.where(far_ok,far,0.))
    # A sharp-to-blurred transition is one hallmark of an orthomosaic seam.
    texture=cv2.GaussianBlur(np.abs(raw-cv2.GaussianBlur(raw,(0,0),1.15)),(0,0),2.)
    texture_samples=texture[yy,xx]
    texture_l=np.median(texture_samples[:,4:11],axis=1)
    texture_r=np.median(texture_samples[:,36:43],axis=1)
    lower=np.minimum(texture_l,texture_r)
    upper=np.maximum(texture_l,texture_r)
    seam=(lower<5.2)&(upper>6.)&(upper>lower*2.10)
    blurry=upper<4.5
    weak=allowed & ~seam & (contrast>=np.where(blurry,12.,8.5)) & (inner>=np.where(blurry,2.,1.))
    strong=allowed & ~seam & (contrast>=np.where(blurry,17.,14.)) & (inner>=2.4)
    ws=np.zeros(skeleton.shape,np.uint8); ss=np.zeros_like(ws)
    ws[y[weak],x[weak]]=1; ss[y[strong],x[strong]]=1
    filt=lambda a: cv2.boxFilter(a,cv2.CV_32F,(21,21),normalize=False,borderType=cv2.BORDER_CONSTANT)
    n=filt(skeleton.astype(np.uint8))
    w=filt(ws)
    s=filt(ss)
    neighborhood=(w>=np.maximum(5.,.49*n))&(s>=np.maximum(3.,.24*n))
    # Local tolerance preserves small intensity interruptions along a reviewed
    # trace, while source-mask gaps and unsupported stretches never get filled.
    nearby=cv2.dilate(ws,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0
    reject=np.zeros(skeleton.shape,bool)
    reject[y[seam],x[seam]]=True
    return skeleton & neighborhood & nearby & valid & ~reject


def runs(mask):
    if not mask.any():
        return
    l,t,w,h=cv2.boundingRect(mask.astype(np.uint8))
    local=mask[t:t+h,l:l+w].astype(np.uint8)
    _,labels,stats,_=cv2.connectedComponentsWithStats(local,connectivity=8)
    for i,(x,y,ww,hh,n) in enumerate(stats[1:],1):
        if n>=MIN_SPAN and max(ww,hh)>=MIN_SPAN:
            yield l+int(x),t+int(y),labels[y:y+hh,x:x+ww]==i
