from __future__ import division

import os
import sys
import cv2
import csv
import tqdm
import pandas as pd
from sort.sort import *
import random
import numpy as np
import time
from matplotlib import pyplot as plt
from re import sub
from utils.utils import *
import fnmatch
import scipy.optimize
from skimage.metrics import structural_similarity as ssim
import time
from tqdm import tqdm




def analyzeSORT(df,threshold,slow_move,delta_overlap):
    vc = df.label.value_counts()
    test = vc[vc > threshold].index.tolist()
    deadboxes = []
    deathspots = []  
    for ID in test:      
        filtval = df['label'] ==ID
        interim = df[filtval]
        interimD = []
        interim2 = np.asarray(interim)
        fill = 0
        deadcount = 0
        alivecount = 0
        isdead = 0
        x1E = interim['x1'].iloc[0]
        y1E = interim['y1'].iloc[0]
        x2E = interim['x2'].iloc[0]
        y2E = interim['y2'].iloc[0]
        for row in reversed(interim2):
            frameNA, x1A, y1A, x2A, y2A, labelA, deltaA, catagoryA, *_ = row
            if fill > 1:
                boxA = [x1A, y1A, x2A, y2A]
                boxB = [x1B, y1B, x2B, y2B]
                deltaA = bb_intersection_over_union(boxA, boxB) 
                if deltaA > delta_overlap:
                    deadcount+=abs(frameNA-frameNB)           
                if deadcount > slow_move:
                    if deadboxes == []:
                        deathspots.append([frameNA, x1A, y1A, x2A, y2A,labelA])                          
                        deadboxes.append([x1A, y1A, x2A, y2A])
                        isdead = 1
                        x1Z = x1A
                        y1Z = y1A
                        x2Z = x2A
                        y2Z = y2A
                    else:
                        notunique = 0
                        for box in deadboxes:
                            #print(box)
                            x1D, y1D, x2D, y2D, *_ = box
                            boxD = [x1D, y1D, x2D, y2D]
                            deltaD = bb_intersection_over_union(boxA, boxD)  
                            if deltaD > 0.2:
                                notunique = 1
                        if notunique == 0:
                            deathspots.append([frameNA, x1A, y1A, x2A, y2A,labelA])  
                            deadboxes.append([x1A, y1A, x2A, y2A])
                            isdead = 1
                            x1Z = x1A
                            y1Z = y1A
                            x2Z = x2A
                            y2Z = y2A
                if isdead==1 and deadcount > slow_move:               
                    boxA = [x1B, y1B, x2B, y2B]
                    boxZ = [x1Z, y1Z, x2Z, y2Z]
                    deltaZ = bb_intersection_over_union(boxA, boxZ) 
                    if deltaZ < 0.4:
                        #alivecount+=abs(frameNA-frameNB)
                        #print(frameNA,frameNB,deltaZ,deadcount,labelA)
                        #if alivecount > 1:
                        deadcount = 1
                        deathspots =  deathspots[:-1] 
                        deadboxes = deadboxes[:-1]
                        isdead=0
                #if deadcount > 5*slow_move:
                   # print("broke cause of long",frameNA,frameNB,deadcount)
                    #break
            frameNB, x1B, y1B, x2B, y2B,labelB, deltaB, catagoryB, *_ = row
            fill +=1                
                
                
    csv_outputs = pd.DataFrame(deathspots, columns = ['frame','x1A', 'y1A', 'x2A', 'y2A','labelA'])
    return(csv_outputs)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if float(boxAArea + boxBArea - interArea) != 0:
    
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou=0
    # return the intersection over union value
    return iou







if __name__ == "__main__":

    CSV_FOLD_PATH = sys.argv[1] #folder of YOLO outputs
    OUT_FOLD_PATH = sys.argv[2]
    
    #ARGS
    threshold = 2 #number of frames a worm has to be tracked in order to be analyzed
    slow_move = 15 #number of frames overlapping by 'delta_overlap' before being called dead or paralyzed (15ish=dead,5=paralyzed)
    delta_overlap = 0.95 #%overlap to be called motionless (.95 for dead, .8 for paralyzed
    max_frame = 5000

    csv_list = os.listdir(CSV_FOLD_PATH)
    csv_list = list(filter(lambda f: f.endswith('.csv'), csv_list))

    print(csv_list)
    csvindex = 0
    #loop through list of CSVs
    for csv_name in csv_list:
        csvindex +=1
        print("starting", csv_name,"which is ",csvindex,"of",len(csv_list))
        start_time = time.time()
        csv_PATH = os.path.join(CSV_FOLD_PATH, csv_name)
        OUT_PATH = os.path.join(OUT_FOLD_PATH, csv_name)
        
        #read csv and reformat
        df = pd.read_csv(csv_PATH,names=('frame', 'x1', 'y1', 'w','h','label'))
        df['x2']=df[['x1','w']].sum(axis=1)
        df['y2']=df[['y1','h']].sum(axis=1)
        df = df[['frame','x1', 'y1', 'x2', 'y2']]
        filtval = df['frame'] < max_frame
        df = df[filtval]
        
        unique = df["frame"].unique()

        #initialize sort tracker and create container       
        mot_tracker1 =Sort(max_age=0, min_hits=0, iou_threshold=0.3)  #see SORT documentation NEEDS TUNING
        sort_outputs = []
        print("sorting")
        
        #sort from end of experiment backwards
        for x in reversed(unique):
            frame = int(x)
            filtval = df['frame'] == x
            boxes_xyxy = np.asarray(df[filtval])[:,1:5]
            track_bbs_ids = mot_tracker1.update(boxes_xyxy)
            for output in track_bbs_ids:
                x1, y1, x2, y2, label, *_ = output
                sort_outputs.append([x.tolist(), x1.tolist(), y1.tolist(), x2.tolist(),y2.tolist(),label.tolist()])
        sort_outputs = pd.DataFrame(sort_outputs)
        sort_outputs.columns = ['frame','x1', 'y1', 'x2', 'y2','label']
        vc = sort_outputs.label.value_counts()
        test = vc[vc > 2].index.tolist()
        sort_outputs = sort_outputs[sort_outputs['label'].isin(test)]
        
        
        #create container dataframes for maximum and minimum frames for each label
        dfmin = pd.DataFrame(columns=['frame', 'x1', 'y1','x2','y2','label'])
        dfmax = pd.DataFrame(columns=['frame', 'x1', 'y1','x2','y2','label'])
        print("creating max/min arrays")
        #for each label, create max and minimum arrays. This is done dumbly feel free to improve via vectorization.
        uniqueTracks = sort_outputs["label"].unique()
        for track in uniqueTracks:
            filtval = sort_outputs['label'] == track
            trackdf = sort_outputs[filtval]
            max_value = trackdf['frame'].max()
            min_value = trackdf['frame'].min()
            trackmin = sort_outputs[(sort_outputs["label"]==track) & (sort_outputs["frame"]==min_value)]
            trackmax = sort_outputs[(sort_outputs["label"]==track) & (sort_outputs["frame"]==max_value)]
            dfmin = dfmin.append(trackmin, ignore_index = True)
            dfmax = dfmax.append(trackmax, ignore_index = True)

        #take necessary arrays and reformat all to correct data type (int)
        #also, remove any SORT label that was observed less than twice. This drastically lowers computation time. 
        #this filter could be removed if the linking step was more improved
        sort2 = sort_outputs
        vc = sort2.label.value_counts()
        test = vc[vc > 2].index.tolist()
        sort2 = sort2[sort2['label'].isin(test)]
        sort2 = sort2.apply(pd.to_numeric)
        sort2 = sort2.apply(np.int64)
        uniqueTracks = sort2["label"].unique()
        dfmax = dfmax.apply(pd.to_numeric)
        dfmax = dfmax.apply(np.int64)
        dfmin = dfmin.apply(pd.to_numeric)
        dfmin = dfmin.apply(np.int64)

        print("linking")

    
        
       
        #this section is kinda a mess but it works
        #goes from labels at the end of the experiment and progresses toward the beginning
        #for a given label (labelmin)
        #gets the miniumum (earliest) frame it was observed, then loops through the maxiumum frames of other labels
        #if there is sufficient overlap between this min and a max, overwrites the 'label' value in all arrays of the max with the 'label' of the min
        #if no overlap is found within a threshold of time (2 days curr), break loop and move to next frame 
        
        #needs tuning and iteration on time and overlap thresholds to figure out optimum
        itersMin = 1
        while itersMin < len(uniqueTracks):
            labelX = uniqueTracks[itersMin]
            #print(labelX)
            wormA = dfmin[dfmin['label'] == labelX]
            #print(wormA)
            frameA = wormA['frame'].iloc[-1]
            x1A = wormA['x1'].iloc[-1]
            y1A = wormA['y1'].iloc[-1]
            x2A = wormA['x2'].iloc[-1]
            y2A = wormA['y2'].iloc[-1]
            labelA = wormA['label'].iloc[-1]
            itersMin +=1
            itersMax = itersMin
            #print(len(uniqueTracks))
            while itersMax < len(uniqueTracks):
                uniqueTracks[uniqueTracks>labelA]
                labelY = uniqueTracks[itersMax]
                #print(labelY)
                wormB = dfmax[dfmax['label'] == labelY]
                frameB = wormB['frame'].iloc[-1]
                x1B = wormB['x1'].iloc[-1]
                y1B = wormB['y1'].iloc[-1]
                x2B = wormB['x2'].iloc[-1]
                y2B = wormB['y2'].iloc[-1]
                labelB = wormB['label'].iloc[-1]
                itersMax+=1
                if frameB < frameA and labelA != labelB:
                    boxA = [x1A, y1A, x2A, y2A]
                    boxB = [x1B, y1B, x2B, y2B]
                    delta = bb_intersection_over_union(boxA, boxB)
                    if abs(frameA-frameB)<5:
                        deltathresh = 0.2
                    elif abs(frameA-frameB)<36:
                        deltathresh = 0.4
                    else:
                        deltathresh = 0.7
                    if delta > deltathresh:
                        #print('changing ',labelB,' to ',labelA,' at frame:',frameB)
                        sort2 = sort2.replace({'label': labelB}, labelA)
                        dfmax =dfmax.replace({'label': labelB}, labelA)
                        dfmin = dfmin.replace({'label': labelB}, labelA)

                        #dfmax = np.where(dfmax == labelB, labelA, dfmax)
                        #dfmin = np.where(dfmin == labelB, labelA, dfmin)
                        uniqueTracks[uniqueTracks==labelB]=labelA
                        #maxOver.append(labelB)
                        break            
                    if frameB < (frameA-144):
                        break



        #print(sort2)
        #reformat output to be accepted into analyze sort
        csv_outputs = pd.DataFrame(sort2)
        csv_outputs.columns = ['frame', 'x1', 'y1', 'x2', 'y2','label']
        csv_outputs['delta'] = 0
        csv_outputs['catagory'] = 'alive'
        
        #analyze for death
        outputs = analyzeSORT(csv_outputs,threshold,slow_move,delta_overlap)
        #print(outputs)
        outputs['expID'] = os.path.basename(csv_PATH).strip('.csv')

        #export and move to next csv file
        pd.DataFrame(outputs).to_csv(OUT_PATH, mode='w', header=True, index=None)
        print('finished in:',time.time()-start_time,'seconds')   
    
    
    
    
    
    
    
    
    
    