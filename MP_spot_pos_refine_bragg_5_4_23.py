 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit

exp_data = np.load('exp_data.npy')

bkg_limit = np.load('bkg_limit.npy')  #These limites define the radius around the peak for the background versus peak regions
peak_limit = np.load('peak_limit.npy')

step = ( peak_limit ) / 4  #This defines the amount of upsampling

x_low_limit = np.load('y_low_limit.npy')  #These limits make sure that the fitting windows don't go outside the 128 x 128 pixel region
x_high_limit = np.load('y_high_limit.npy')

y_low_limit = np.load('x_low_limit.npy')
y_high_limit = np.load('x_high_limit.npy')

def plane(xdata, mx, my, C):  #2D plane used to fit background

    xdata = np.array(xdata)
    intensities = np.zeros(len(xdata))    
    intensities = C + xdata[:,0]*mx + xdata[:,1]*my   
    return intensities

#define refinment function for CDW peaks
def spot_pos_refine(good_spot, T, y_guess, x_guess):   #T labels the specific diffraction pattern (I convert from 4d to 3d dataset during analysis) and the y_guess and x_guess are the initial guesses for the peak position
    
    if good_spot == 1:
        
        #Load the diff pattern
        pattern = exp_data[int(T),:,:] + 0

        x_int = round(x_guess) #Round the guess to the nearest pixel to begin with
        y_int = round(y_guess)
    
        #Define the size of the initial (large) fitting window
        FW_half = 2 #total size of fitting window = FW_half*2 + 1
        FW = pattern[ x_int - FW_half : x_int + FW_half + 1 , y_int - FW_half : y_int + FW_half + 1 ]

        #Find the local maximum in the FW, and return the shifts needed to bring this pixel to the FW center
        max_shift = np.unravel_index(np.argmax(FW), FW.shape)
    
        #Update x and y guess based on the local max
        x_updated = (x_int + (max_shift[0] - FW_half))
        y_updated = (y_int + (max_shift[1] - FW_half))
    
        #Make sure that the updated peak position isn't outside the detector limits
        if x_updated > x_low_limit and x_updated < x_high_limit and y_updated > y_low_limit and y_updated < y_high_limit:                

            x_PAD = np.arange(0,128,1)
            y_PAD = np.arange(0,128,1)

            f = interpolate.interp2d(x_PAD, y_PAD, pattern, kind='linear')
                
            x_FW = np.arange( x_updated - bkg_limit , x_updated + bkg_limit ,  step)  
            y_FW = np.arange( y_updated - bkg_limit , y_updated + bkg_limit ,  step)  
                
            #FW = fitting window
            FW = f( y_FW, x_FW )  #Upsample the data near the peak using linear interpolation. Upsampling is defined by 'step'
            
            xdata_all = []
            xdata_bkg = []
            intensities_bkg = []

            #Iterate through all points in fitting window, and determine if they are in the background region
            for i in range (0, len(x_FW)):
                for j in range (0, len(y_FW)):
                        
                    xdata_all.append([ x_FW[i], y_FW[j] ])
                        
                    distance = ( ( x_FW[i] - x_updated )**2  +  ( y_FW[j] - y_updated )**2 )**0.5
                        
                    if distance > peak_limit and distance < bkg_limit:
                            
                        xdata_bkg.append([ x_FW[i], y_FW[j] ])
                        intensities_bkg.append( FW[i,j] )
            
            
            #Fit background datapoints with 2D plane
            p0 = [ np.mean(intensities_bkg), 0 , 0]
            popt,pcov = curve_fit(plane, xdata_bkg, intensities_bkg, p0 = p0)

            bkg_fitted = plane(xdata_all, *popt)
            bkg_fitted = np.reshape(bkg_fitted, (len(x_FW),len(y_FW)))
                
            #Subtract the fitted background plane from the fitting window
            FW = FW - bkg_fitted
            
            #Discared all datapoints outside of 'peak region'
            for i in range (0, len(x_FW)):
                for j in range (0, len(y_FW)):
                                                
                    distance = ( ( x_FW[i] - x_updated )**2  +  ( y_FW[j] - y_updated )**2 )**0.5
                        
                    if distance > peak_limit:
                            
                        FW[i,j] = 0
                
            #Normalized data to integral unity
            FW = FW / np.sum(FW)
            
            #Perform COM calculation along x and y axes independently
            FW_integrated_x = np.sum(FW, axis = 1) #average along y-axis
            COM_x = np.dot(x_FW, FW_integrated_x) #Compute COMx
            FW_integrated_y = np.sum(FW, axis = 0) #average along y-axis  
            COM_y = np.dot(y_FW, FW_integrated_y) #Compute COMx 

            #Update peak position based on first COM calculation
            x_refined = COM_x
            y_refined = COM_y
            
            #Check to make sure the refined peak positions are still within 128 x 128 pixel limits
            if x_refined > x_low_limit and x_refined < x_high_limit and y_refined > y_low_limit and y_refined < y_high_limit:   
                
                #Repeat everything above, but with refined starting point
                x_FW = np.arange( x_refined - bkg_limit , x_refined + bkg_limit ,  step)  
                y_FW = np.arange( y_refined - bkg_limit , y_refined + bkg_limit ,  step)  
                
                FW = f( y_FW, x_FW )
            
                xdata_all = []
                xdata_bkg = []
                intensities_bkg = []

                for i in range (0, len(x_FW)):
                    for j in range (0, len(y_FW)):
                        
                        xdata_all.append([ x_FW[i], y_FW[j] ])
                        
                        distance = ( ( x_FW[i] - x_refined )**2  +  ( y_FW[j] - y_refined )**2 )**0.5
                        
                        if distance > peak_limit and distance < bkg_limit:
                            
                            xdata_bkg.append([ x_FW[i], y_FW[j] ])
                            intensities_bkg.append( FW[i,j] )
            
            
                p0 = [ np.mean(intensities_bkg), 0 , 0]
                popt,pcov = curve_fit(plane, xdata_bkg, intensities_bkg, p0 = p0)
                
                bkg_std = np.std( plane(xdata_bkg, *popt) - intensities_bkg )
                bkg_avg = np.mean( intensities_bkg )
                
                bkg_fitted = plane(xdata_all, *popt)
                bkg_fitted = np.reshape(bkg_fitted, (len(x_FW),len(y_FW)))
                
                FW = FW - bkg_fitted
            
                for i in range (0, len(x_FW)):
                    for j in range (0, len(y_FW)):
                                                
                        distance = ( ( x_FW[i] - x_refined )**2  +  ( y_FW[j] - y_refined )**2 )**0.5
                        
                        if distance > peak_limit:
                            
                            FW[i,j] = 0
                
                total_intensity = np.sum(FW)
                max_intensity = np.max(FW)
                
                FW = FW / np.sum(FW)
            
                #Perform COM calculation along x and y axes independently
                FW_integrated_x = np.sum(FW, axis = 1) #average along y-axis
                COM_x = np.dot(x_FW, FW_integrated_x) #Compute COMx
                FW_integrated_y = np.sum(FW, axis = 0) #average along y-axis  
                COM_y = np.dot(y_FW, FW_integrated_y) #Compute COMx 

                x_final = COM_x
                y_final = COM_y

                              
                distance = ((y_final - y_guess)**2 + (x_final - x_guess)**2 )**0.5
                spread = ( np.max(FW) / np.sum(FW) ) 


        
                return 1, y_final, x_final, total_intensity, distance, spread, max_intensity, bkg_avg, bkg_std
            
            else:
        
                return 0, 0, 0, 0, 0, 0, 0, 0, 0
    
        else:
        
            return 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    else:
        
        return 0, 0, 0, 0, 0, 0, 0, 0, 0


     
        

        
        
        