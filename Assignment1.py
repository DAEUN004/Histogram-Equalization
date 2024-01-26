##Please type main(Case1) for Question 1
##main(Case3) for Question 3
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

images_list = ["sample01.jpg" ,"sample02.jpeg","sample03.jpeg","sample04.jpeg","sample05.jpeg",
                 "sample06.jpg","sample07.jpg","sample08.jpg"]

for image in images_list:
    ##Original Image
    img = cv.imread(image) 
    img_array = np.asarray(img)
    colours = cv.split(img_array)

    #To get the dimension of the given image
    dimensions = img.shape
    H = dimensions[0]
    W = dimensions[1]

    def histequ(hist):
        #Normalizing histogram
        cdf = hist.cumsum()
        #NewIntensity = ((Intensity - minimum)/(maximum - minimum))*255
        s_k = (cdf - cdf.min())/(cdf.max() - cdf.min()) * 255
        s_k = s_k.astype('uint8')
        return s_k

    def histequ_clip(hist):
        #Clipped Histogram Equalization for part 3
        hist = np.clip(hist,a_min=0, a_max = np.mean(hist)*1.5)
        return histequ(hist)


    def img_processing(section,function):
        #section: 
        #function: histequ/histequ_clip
        H = section.shape[0]
        W = section.shape[1]

        img_array = np.asarray(section)
        #Splitting the image into three channels
        colours = cv.split(img_array)
        
        final_image = []
        
        #Histogram Equalization performed for each channel
        for index, colour in enumerate(colours):

            flatten_image = colour.flatten()
            hist = cv.calcHist([section], [index], None, [256], [0,256])
            img_new = function(hist)[flatten_image]
            img_new = np.reshape(img_new, (H,W,1))
            final_image.append(img_new)
        
        img_new = cv.merge([final_image[0], final_image[1], final_image[2]])
        return img_new
    
    
    #QUESTION 1
    img_new = img_processing(img,histequ) 
    ##Validation
    equ_b = cv.equalizeHist(colours[0])
    equ_g = cv.equalizeHist(colours[1])
    equ_r = cv.equalizeHist(colours[2])
    img_equ = cv.merge([equ_b, equ_g, equ_r])

    #QUESTION 3: Clipped Histogram Equalization
    img_new_clip = img_processing(img,histequ_clip)

    #Question 3: Sectional Histogram Equalization
    ##Dividing image into eight parts
    half_width = H//2
    half_half_width = half_width//2
    half_height = W//2

    section_1 = img[:half_height, :half_half_width]
    section_2 = img[:half_height, half_half_width:half_width]
    section_3 = img[:half_height, half_width:half_width + half_half_width]
    section_4 = img[:half_height, half_width + half_half_width:]
    section_5 = img[half_height:, :half_half_width]
    section_6 = img[half_height:, half_half_width:half_width]
    section_7 = img[half_height:, half_width:half_width + half_half_width]
    section_8 = img[half_height:, half_width + half_half_width:]
    

    #Processing each image section
    img_section_1 = img_processing(section_1,histequ_clip)
    img_section_2 = img_processing(section_2,histequ_clip)
    img_section_3 = img_processing(section_3,histequ_clip)
    img_section_4 = img_processing(section_4,histequ_clip)
    img_section_5 = img_processing(section_5,histequ_clip)
    img_section_6 = img_processing(section_6,histequ_clip)
    img_section_7 = img_processing(section_7,histequ_clip)
    img_section_8 = img_processing(section_8,histequ_clip)

    #Combing each section into one piece
    im_h_1 = cv.hconcat([img_section_1,img_section_2,img_section_3,img_section_4])
    im_h_2 = cv.hconcat([img_section_5,img_section_6,img_section_7,img_section_8])

    img_partial = cv.vconcat([im_h_1,im_h_2])

    

    def main(case_number):
        #Case 1: Question 1, Images Before and After Histogram Equalization
        #Case 2: Question 1, Histograms Before and After Histogram Equalization
        #Case 3: Question 3, Images Before and After Improved Histogram Equalization
        return case_number()

    
    def Case1():
        #img: Original image
        #im_new: After Histogram Equalization
        #im_equ: For validation which used OpenCV inbuilt histogram equalization

        plt.figure(figsize=(10,5))
        plt.subplot(231)
        plt.imshow(img)
        plt.title("Original Image", loc= 'center',fontsize = 8)
        
        plt.subplot(232)
        plt.imshow(img_new)
        plt.title("Histogram Equalization with Own Algorithm", loc= 'center',fontsize = 8)

        plt.subplot(233)
        plt.imshow(img_equ)
        plt.title("Validation", loc= 'center',fontsize = 8)
        
        plt.show()
        
    def Case2():
        plt.figure()
        colors = ('b','g','r')
        for i, col in enumerate(colors):
            
            hist_before = cv.calcHist([img],[i],None,[256],[0,256])
            plt.subplot(131)
            plt.plot(hist_before, color = col)
            plt.title("Before Histogram Equalization", loc= 'center',fontsize = 8)


            hist_after = cv.calcHist([img_new],[i],None,[256],[0,256])
            plt.subplot(132)
            plt.plot(hist_after, color = col)
            plt.title("After Histogram Equalization", loc= 'center',fontsize = 8)

            hist_after_validation = cv.calcHist([img_equ],[i],None,[256],[0,256])
            plt.subplot(133)
            plt.plot(hist_after_validation, color = col)
            plt.title("After Histogram Equalization", loc= 'center',fontsize = 8)
            
        plt.show()
        
    
    def Case3():
        plt.figure(figsize=(10,15))
        plt.subplot(221)
        plt.imshow(img)
        plt.title("Original Image", loc= 'center',fontsize = 8)

        plt.subplot(222)
        plt.imshow(img_new)
        plt.title("Histogram Equalization with Own Algorithm", loc= 'center',fontsize = 8)

        plt.subplot(223)
        plt.imshow(img_new_clip)
        plt.title("Clipped Histogram Equalization", loc= 'center',fontsize = 8)
        
        plt.subplot(224)
        plt.imshow(img_partial)
        plt.title("Sectional Histogram Equalization", loc= 'center',fontsize = 8)


        plt.show()
    
    
    
    
    
    main(Case3)
        


   



    


    



