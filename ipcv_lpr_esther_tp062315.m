close all;
clear all;
clc;

%%%% License Plate Segmentation 
% Step 1: Read image 
%img_ori = imread('c1.png'); %1,4,7
img_ori = imread('c3.jpg');
img = rgb2gray(img_ori);
subplot(3,3,1), imshow(img), title('1. Grayscale');


% Step 2: Detect entire cell using Prewitt and then threshold % ori = prewitt/0.5
% Sobel, Canny, Prewitt, Roberts (0.5 is the treshold level) 
bw = edge(img,'Prewitt',(graythresh(img)*0.5));
subplot(3,3,2), imshow(bw), title('2. Edge Detection');


% Step 3: Dilate the image (length,angle)
% Structuring element 
se90 = strel('line',1,90); % vertical line, smaller value because don't want other parts to get big
se0 = strel('line',35,0); % horizontal line, higher value so that can get bigger blob horizontally across the numbers
% Dilation 
bwsdil = imdilate(bw, [se90 se0]);
subplot(3,3,3), imshow(bwsdil), title('3. Dilated gradient mask');

% Step 4: Fill holes 
% 4 connectivity , cross shape
bwdil_fill = imfill(bwsdil, 'holes');
subplot(3,3,4), imshow(bwdil_fill), title('4. Filled holes');

% Step 5: Remove connected objects on border 
bwnobord = imclearborder(bwdil_fill,8); 
subplot(3,3,5), imshow(bwnobord), title('5. Cleared border image');

% Step 6: Repeated Erosion to Remove Noise
seD = strel('square',5);
bwfinal = imerode(bwnobord, seD);
bwfinal = imerode(bwfinal,seD); 
subplot(3,3,6), imshow(bwfinal), title ('6. Repeated Erosion');

% Step 7: Retain Car Plate Segment only 
% Get area/pixel of biggest blob 
stats = regionprops(bwfinal, 'Area');
biggest_area = max( [stats.Area] );

% Remove components other than the biggest blob
bwfinal = bwareaopen(bwfinal,(biggest_area-20)); 
subplot(3,3,7), imshow(bwfinal), title ('7. Segment License Plate Area Binary Mask');

% Step 8: Dilate to make it bigger(using rectangle)Binary Mask 
se = strel('rectangle',[20,40]);
bwfinal = imdilate(bwfinal, se);
subplot(3,3,8), imshow(bwfinal), title('8. License Plate Area Dilated Binary Mask');

% Step 9: Crop based on binary mask instead of bounding box 
% label and rotate image
[labeledImage] = bwlabel(bwfinal);
measurements = regionprops(labeledImage, 'Orientation');
rotatedOri = imrotate(img, -measurements(1).Orientation);
rotatedBW = imrotate(bwfinal, -measurements(1).Orientation);

% Make bounding box.
rot_measurements = regionprops(rotatedBW, 'BoundingBox');
boundingBox = rot_measurements.BoundingBox;
width = boundingBox(3);
height = boundingBox(4);
xLeft = boundingBox(1);
yTop = boundingBox(2);
boundingBox2 = [xLeft+15, yTop, width-15, height];
new_crop = imcrop(rotatedOri, boundingBox2);

subplot(3,3,9),imshow(new_crop),title('9. Cropped License Plate');

%%%% Character Segmentation 
% Step 10: Binarize cropped image 
new_cropbw = imbinarize(new_crop);
figure(2),subplot(5,1,1),imshow(new_cropbw),title('1. Binarize Cropped License Plate');

% % Step 11: Clear borders
new_cropbw = imclearborder(new_cropbw,8); 
figure(2),subplot(5,1,2), imshow(new_cropbw), title('2. Cleared borders');

% Step 12: if num/characters not white then change from black to white 
% fill holes
crop_fill = imfill(new_cropbw, 'holes'); % so that we get the are of the plate

% count number of black and white pixels in image
[b,w] =countBW(crop_fill);
% nWhite = sum(crop_fill(:));
% nBlack = numel(crop_fill) - nWhite;

[row,col] = size(new_cropbw);
bw_convert = new_cropbw;

if w > b
    for i = 1:row
        for j = 1:col
            if bw_convert(i,j) > 0
                bw_convert(i,j) = 0;
            else
                bw_convert(i,j) = 1;
            end
            
        end
        
    end
else
    bw_convert = bw_convert;
end

figure(2),subplot(5,1,3),imshow(bw_convert),title('3. Convert those with black letters to white else remain');

% % Step 13: Clear borders
bw_convert_clean = imclearborder(bw_convert,8); 
figure(2),subplot(5,1,4), imshow(bw_convert_clean), title('4. Cleared border Image 2');

% Step 14: Remove small blobs from image 
bw_final_clean = bwareaopen(bw_convert_clean,20); 
figure(2),subplot(5,1,5), imshow(bw_final_clean), title('5. Remove unwanted objects');


% Step 15a: Character Segmentation & Recognition
[labeledImage, numBlobs] = bwlabel(bw_final_clean);
props = regionprops(labeledImage, 'BoundingBox');
for k = 1 : numBlobs
    thisBB = props(k).BoundingBox;
    thisImage{k} = imcrop(bw_final_clean, thisBB);
    results{k} = ocr(thisImage{k},'TextLayout','Block');
    %results{k} = ocr(thisImage{k},'TextLayout','Character');
    num_plate{k} = results{k}.Text;
    %figure, imshow(thisImage{k});
    drawnow;

end
% Convert characters to string 
num_plate = convertCharsToStrings(num_plate);
num_plate = regexprep(num_plate,'\n+','');

% Step 15b: Character recognition without segmentation 
results2 = ocr(bw_final_clean,'TextLayout','Block');
num_plate2 = results2.Text;

%%% Display Results 
disp(num_plate);
disp(num_plate2);


%%%%%%%%%%%%%%%%%%%%%%%%%% Optional %%%%%%%%%%%%%%%%%%%%%%%%%
%%% BF Scoring 
% 1) for num plate segmentation 
mask = false(size(img));
mask(25:end-25,25:end-25) = true;
BW = activecontour(img, mask, 300);

%Read the ground truth segmentation.
BW_groundTruth = bwfinal;

%Compute the BF score of the active contours segmentation against the ground truth.
similarity = bfscore(BW, BW_groundTruth);

%Display the masks on top of each other. Colors indicate differences in the masks.
figure
imshowpair(BW, BW_groundTruth)
title(['BF Score = ' num2str(similarity)])

%%% Function 
function [black,white]=countBW(Image)
  black=length(Image(Image==0));
  white=length(Image(Image==1));
end