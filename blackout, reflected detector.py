import cv2
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

#opticdisc demo version
def _opticdisc(img):
    
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    r,thr= cv2.threshold(imgray,170,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow('',thr)

    return thr

def get_imgsize(img):
    height = img.shape[0] 
    width = img.shape[1]

    return width,height

#parameter는 이미지, 위 꼭지점 높이  비율, 위 꼭지점 높이 비율...
def Gray_ROI(img,h_top,h_bottom,w_left,w_right):
    width, height = get_imgsize(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    roi = gray[int(height * h_top) : int(height * h_bottom) , int(width * w_left) : int(height * w_right)]

    
    
    roiwidth, roiheight = get_imgsize(roi)
    
    return roi, roiwidth, roiheight
    

#1번째로 검사
def check_blackout_img(img):
    
    
    
    roi,width,height = Gray_ROI(img,0.2,0.8,0,1)
    #roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

    plt.hist(roi.ravel(), 256,[0,256])
    plt.show()
    
    #cv2.imshow('b',roi)
    #리스트 초기화
    hist = [0]*256
    #이미지 각각의 밝기값 저장
    for x in range(width):
        for y in range(height):
            if roi[y][x] > 20:
                hist[roi[y][x]] = hist[roi[y][x]] + 1
            
                #hist[img.item(y,x,0)] = hist[img.item(y,x,0)] + 1
            
    #blackout 검사   
    histdark=0
    histnormal=0
    histbright=0

    #가중치
    weight=1000
    weight_range=15
    
    #가중치에 따른 히스토그램 분석
    for i in range(2,50):
        if i <= weight_range:#~15
            histdark = histdark + (hist[i]*weight)#2~15
        elif i> weight_range and i <= weight_range + 7: #16~23 가중치:100
            histdark = histdark + (hist[i]*(weight*0.1))
        elif i> weight_range+7 and i <= weight_range + 15:#24~30 가중치:10
            histdark = histdark + (hist[i]*(weight*0.01))
        else:
            histdark = histdark + hist[i]
    for i in range(51,254):
        histnormal = histnormal + hist[i]
    

    hist_max = max(histdark,histnormal)

    #print('histdark:',histdark,'histnormal:',histnormal)
    
    if hist_max == histdark:
        #print('blackout img')
        return True
    else:
        #print('none blackout img')
        return False

def check_blur(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)
    f1, thr = cv2.threshold(gray,80,255,cv2.THRESH_TOZERO)
    cv2.imshow('thr',thr)

def lumin_avg(roi,width,height):
    count = 0
    _sum = 0
    for x in range(0,width-1):
        for y in range(0,height-1):
            if roi[y][x] > 20:
                _sum = _sum + roi[y][x]
                count = count + 1
    if count != 0:        
        avg = _sum / count
    else:
        avg =0
    return avg

def check_reflected(img):    

    ## mmask는 빛반사영역, nmask는 빛반사영역 바로 옆 영역
    
    #원본이미지와 mask를 합성
    mask = cv2.imread('3mmask.jpg')
    img_and_mask = cv2.bitwise_and(img, mask)
    sect_mask=[]
    #cv2.imshow('img_and_mask',img_and_mask)
    
    #[0]left,top [1]right,top [2]left,bottom [3]right,bottom
    sect_roi,width,height = Gray_ROI(img_and_mask,0.2, 0.4, 0, 0.5)    
    sect_mask.append(lumin_avg(sect_roi, width, height))    
    #cv2.imshow('sect_roi1',sect_roi)
    
    sect_roi,width,height = Gray_ROI(img_and_mask,0.2, 0.4, 0.5, 1)
    sect_mask.append(lumin_avg(sect_roi, width, height))    
    #cv2.imshow('sect_roi2',sect_roi)
    
    sect_roi,width,height = Gray_ROI(img_and_mask,0.6, 0.8, 0, 0.5)    
    sect_mask.append(lumin_avg(sect_roi, width, height))
    #cv2.imshow('sect_roi3',sect_roi)
    
    sect_roi,width,height = Gray_ROI(img_and_mask,0.6, 0.8, 0.5, 1)
    sect_mask.append(lumin_avg(sect_roi, width, height))
    #cv2.imshow('sect_roi4',sect_roi)
    
    #nmask(빛 셈 현상이 일어나는 구역 바로옆 마스크)
    sect_nmask=[]
    nmask = cv2.imread('3nmask.jpg')
     
    
    opticmask = _opticdisc(img)
    opticmask = cv2.cvtColor(opticmask,cv2.COLOR_BGR2RGB)
    
    nmask_and_optic = cv2.bitwise_and(nmask, opticmask)
    #cv2.imshow('n_and_o',nmask_and_optic)
    
    img_and_optic = cv2.bitwise_and(img, nmask_and_optic)   
    #cv2.imshow('img_and_o',img_and_optic)
    
    sect_roi,width,height = Gray_ROI(img_and_optic,0.2, 0.4, 0, 0.5)    
    sect_nmask.append(lumin_avg(sect_roi, width, height))
    #cv2.imshow('sect_roi5',sect_roi)
   
    sect_roi,width,height = Gray_ROI(img_and_optic,0.2, 0.4, 0.5, 1)
    sect_nmask.append(lumin_avg(sect_roi, width, height))    
    #cv2.imshow('sect_roi6',sect_roi)
    
    sect_roi,width,height = Gray_ROI(img_and_optic,0.6, 0.8, 0, 0.5)    
    sect_nmask.append(lumin_avg(sect_roi, width, height))
    #cv2.imshow('sect_roi7',sect_roi)
    
    sect_roi,width,height = Gray_ROI(img_and_optic,0.6, 0.8, 0.5, 1)
    sect_nmask.append(lumin_avg(sect_roi, width, height))
    #cv2.imshow('sect_roi8',sect_roi)
    
    #보정상수
    C=10
    #print(sect_mask)
    #print(sect_nmask)
    #reflected
    for i in range(len(sect_mask)):
        if sect_mask[i] > sect_nmask[i] + C:
            #print('reflected')
            return True
        else:
            pass
    #print('none reflected')
    return False
                
######mask된 곳 바로 옆에 영역과 차이가 발생하면 reflected.

def circle_lumin_avg(img):
    #원본사진으로부터 ROI, 관심영역 추출, 관심영역 너비, 높이
    roi,width,height = Gray_ROI(img)    

    #원의 방정식(roi의 중앙에서 반지름이 r인 원 영역 생성)
    x1 = int(width / 2) 
    y1 = int(height / 2)
    r = 10
    _sum=0
    count=0
    

    #x의 범위는 원의중심-r ~ 원의중심 + r
    for x in range(int(x1-r),int(x1+r)+1):
        for y in range(int(y1-r),int(y1+r)+1):

            #만약 원의 안쪽으로 들어오는 픽셀 일 경우
            if ((x-x1)*(x-x1) + (y-y1)*(y-y1)) <= r*r:                
                #print(count,'. (',(x-x1),(y-y1),')','=>',x,y)

                #넓이가 아니라 픽셀 수이므로 넓이와 픽셀 수(count)와는 상관없음.
                count = count + 1                
                _sum = _sum + roi[y][x]
                
    #원영역안의 픽셀 값 평균 구하기
    avg = _sum / count
    #print(avg)

    return avg
 
def makemask(img):
    width, height = get_imgsize(img)
    print(height,width)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    for x in range(width-1):
        for y in range(height-1):
            if gray[y][x] > 30:
                gray[y][x] = 255
            else:
                gray[y][x] = 0

    return gray
                
         

if __name__ == "__main__":

    count = 1    
    
    goodFolder = "/Users/user/Desktop/schuhd2/good/"
    badFolder = "/Users/user/Desktop/schuhd2/bad/bla,ref/"
    mixFolder = "/Users/user/Desktop/schuhd2/mix/"

    changeFolder = "/Users/user/Desktop/schuhd2/bad//bla,ref/1/"

    filesArray = [x for x in os.listdir(changeFolder) if os.path.isfile(os.path.join(changeFolder,x))]#경로 수정시 여기 수정

    
    destinationFolder_P = "/Users/user/Desktop/schuhd2/outputP/"
    destinationFolder_N = "/Users/user/Desktop/schuhd2/outputN/"

    if not os.path.exists(destinationFolder_P):
        os.mkdir(destinationFolder_P)
    if not os.path.exists(destinationFolder_N):
        os.mkdir(destinationFolder_N)

    for file_name in filesArray:
        file_name_no_extension = os.path.splitext(file_name)[0]

        img = cv2.imread(changeFolder+'/'+file_name)#경로 수정시 여기 수정

        print(count,'.',file_name)
        
        count = count+1
        check_b = check_blackout_img(img)
        check_r = check_reflected(img)
        #print('')

        
        if check_b == False and check_r == False:
            cv2.imwrite(destinationFolder_P+file_name_no_extension+".jpg",img)
        
        else:
            cv2.imwrite(destinationFolder_N+file_name_no_extension+".jpg",img)
  

#img = cv2.imread('n2.jpg')
#mask = cv2.imread('3mask.jpg')
#check_blackout_img(img)
#check_blur(img)
#mmask = makemask(mask)
#cv2.imwrite('3mmask.jpg', mmask)
#check_reflected(img)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#print(gray[300][300])
    
